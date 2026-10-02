"""(e) The vLLM backend computes what `Judge` computes. CPU only, no vLLM needed.

The vLLM-facing modules are thin; everything they decide is checked here:

* rows: the marker-delimited prompts carry exactly the `layout_rows` rows --
  same split, same tokens, same positions -- and special-token strings inside
  the text are split into plain text instead of breaking the row;
* the step path: `kernel.make_plan`, `kernel.kv_write` (vLLM writes a step's
  K/V before attention), `kernel.run_attention`, `kernel.readout_plan`,
  `kernel.pool_margins`, with what vLLM would supply simulated -- block tables,
  slot mappings, a KV cache in vLLM's block-major layout seen through its
  logical view, prefix hits of whole 16-token blocks, and blocks registered at
  allocation, so the rows of one cold request share the state within a step.

Margins are compared with `Judge.margins_packed` on a small random T5 (fast,
with a wide relative-bias table so that a wrong bucket or mask shows) and on
the exported model in float32.
"""
from __future__ import annotations

import json
import math

import pytest
from conftest import load_cases, record

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from torch import nn  # noqa: E402

from frida_decisions.base import BaseJudge  # noqa: E402
from frida_decisions.packing import layout_rows  # noqa: E402
from frida_decisions.protocol import aggregate, decision  # noqa: E402
from frida_decisions.vllm_backend import kernel  # noqa: E402
from frida_decisions.vllm_backend.rows import (MARK, PREFIX, CONTENT_BREAKERS, build_rows,  # noqa: E402
                                               check_prompt, parse_row, tokenize_rows)

BS = 16          # vLLM's block size
TOL = 2e-5


# ----------------------------------------------------------------- simulated vLLM
class Paged:
    """Per-layer caches in vLLM's physical order [blocks, block_size, heads, 2D],
    handed out through the logical [blocks, heads, block_size, 2D] view, as vLLM
    does. Filled with NaN: a key read from anywhere it was not written poisons
    the margins."""

    def __init__(self, enc, num_blocks: int):
        h, d = enc.n_heads, enc.d_kv
        self.raw = [torch.full((num_blocks, BS, h, 2 * d), float("nan")) for _ in enc.blocks]
        self.views = [r.permute(0, 2, 1, 3) for r in self.raw]
        self.next = 1                        # block 0 is vLLM's null block
        self.registry: dict[tuple, int] = {}  # (prefix tokens) -> block: stands in for block hashes

    def allocate(self, ids: list[int], hit_blocks: int) -> list[int]:
        """Blocks for a prompt; the first `hit_blocks` come from the registry."""
        table = []
        for b in range(-(-len(ids) // BS)):
            key = tuple(ids[:(b + 1) * BS])
            if b < hit_blocks:
                table.append(self.registry[key])
                continue
            table.append(self.next)
            if len(key) == (b + 1) * BS:     # full blocks are registered at allocation
                self.registry.setdefault(key, self.next)
            self.next += 1
        return table

    def longest_hit(self, ids: list[int]) -> int:
        """vLLM: whole registered blocks, at most len - 1 tokens."""
        n = 0
        while (n + 1) * BS <= len(ids) - 1 and tuple(ids[:(n + 1) * BS]) in self.registry:
            n += 1
        return n


def blocks_for(*requests) -> int:
    return 1 + sum(-(-len(row.ids) // BS) for rows in requests for row in rows)


@torch.inference_mode()
def step(enc, paged: Paged, prompts: list[list[int]], hits: list[int] | None = None,
         allow_rejected: bool = False):
    """One vLLM step over several prompts. Returns per-prompt margins and the hits used."""
    # vLLM admits requests one by one: look up, then allocate (which registers the
    # new full blocks), so a later prompt of the same step can hit an earlier one.
    tables, used = [], []
    for i, prompt in enumerate(prompts):
        used.append(hits[i] if hits is not None else paged.longest_hit(prompt))
        tables.append(paged.allocate(prompt, used[-1]))
    ids, qsl, seq, slots = [], [0], [], []
    for p, h, t in zip(prompts, used, tables):
        c = h * BS
        ids += p[c:]
        qsl.append(len(ids))
        seq.append(len(p))
        slots += [t[pos // BS] * BS + pos % BS for pos in range(c, len(p))]
    width = max(len(t) for t in tables)
    block_table = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32)
    plan = kernel.make_plan(enc, ids, qsl, seq, block_table, BS, "cpu")
    bad = [(i, r.parsed.error) for i, r in enumerate(plan.rows) if not r.parsed.valid]
    assert allow_rejected or not bad, f"the model would compute these rows in isolation: {bad}"
    x = enc.embed(torch.tensor(ids))
    slot_mapping = torch.tensor(slots)
    for layer, block in enumerate(enc.blocks):
        q, k, v = block.qkv(x)
        kernel.kv_write(paged.views[layer], k, v, slot_mapping)          # vLLM writes first
        out = torch.empty_like(q)
        kernel.run_attention(plan, q, k, v, paged.views[layer], out)
        x = block.finish(x, out.reshape(x.shape[0], -1))
    hidden = kernel.rms_norm(x, enc.final_ln, enc.eps)
    rp = kernel.readout_plan(prompts, [len(p) - h * BS for p, h in zip(prompts, used)],
                             qsl[:-1], enc.embed.weight.shape[0])
    assert allow_rejected or not rp[4], rp[4]
    assert set(rp[4]) == {i for i, _ in bad}, (rp[4], bad)     # the pooler rejects exactly those
    return kernel.pool_margins(hidden, enc.head, rp), used


def margins_of(rows, per_row, n: int) -> torch.Tensor:
    out = torch.full((n,), float("nan"))
    for row, m in zip(rows, per_row):
        out[row.candidates] = m
    return out


def rows_of(judge, request, nonce=None):
    """Candidates and backend rows of a request, as the IO processor builds them."""
    _, candidates, tok = judge.compile(request)
    same, split = tokenize_rows(candidates, judge.text, judge.config, judge.state_max)
    assert same == tok and not split
    return candidates, build_rows(tok, judge.config, vocab(judge), nonce)


def vocab(judge) -> int:
    return json.loads((judge.folder / "config.json").read_text(encoding="utf-8"))["vocab_size"]


def flat_encoder(t5, head: nn.Linear, share: bool = False) -> kernel.FlatEncoder:
    """The backend's encoder over the very weights `Judge` runs. `share=True`
    points its parameters at the same tensors: one copy of the model in RAM."""
    cfg = t5.config
    dtype = t5.shared.weight.dtype
    wo = t5.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype
    names = None
    if not share:
        enc = kernel.FlatEncoder(cfg, dtype=dtype, wo_dtype=wo)
        loaded = enc.load_hf(t5.state_dict().items(), head=dict(head.state_dict()))
    else:
        with torch.device("meta"):
            enc = kernel.FlatEncoder(cfg, dtype=dtype, wo_dtype=wo)
        sources = dict(t5.state_dict())
        sources.update({f"head.{k}": v for k, v in head.state_dict().items()})
        loaded = set()
        for name, tensor in sources.items():
            local = kernel.hf_to_local(name)
            if local is None or local in loaded:
                continue
            module_name, _, attr = local.rpartition(".")
            module = enc.get_submodule(module_name) if module_name else enc
            assert getattr(module, attr).shape == tensor.shape, local
            setattr(module, attr, nn.Parameter(tensor, requires_grad=False))
            loaded.add(local)
    names = {n for n, _ in enc.named_parameters()}
    assert loaded == names, sorted(names - loaded)[:5]
    return enc.eval()


# ----------------------------------------------------------------- rows
@pytest.fixture(scope="module")
def text_judge(model_dir):
    """Request compiling and tokenizing only (`BaseJudge` has no model)."""
    return BaseJudge(model_dir, state_max=384)


def test_rows_carry_the_packing_layout(text_judge, cases):
    judge = text_judge
    for case in cases:
        candidates, rows = rows_of(judge, case["request"])
        _, _, tok = judge.compile(case["request"])
        layout = layout_rows(tok, len(tok.state), judge.config.max_options_per_row,
                             judge.config.max_row_tokens)
        assert len(rows) == len(layout), case["name"]
        first = 0
        for row, ref in zip(rows, layout):
            assert row.candidates == list(range(first, first + len(ref.options)))
            first += len(ref.options)
            parsed, error = check_prompt(row.ids, 0, vocab(judge))
            assert not error, (case["name"], error)
            content = [i for i, kind in enumerate(parsed.kind) if kind != MARK]
            assert [row.ids[i] for i in content] == tok.state + ref.ids, case["name"]
            assert [parsed.pos[i] for i in content] == list(range(len(tok.state))) + ref.positions
            assert parsed.state_len == len(tok.state)
            assert len(parsed.options) == len(ref.options)
        assert first == len(candidates)


def test_special_token_strings_are_split(text_judge):
    judge = text_judge
    request = {"state": "Текст с <s> и </s> внутри, и даже <mask>.",
               "questions": {"q": {"type": "choice", "instructions": "Что значит <pad>?",
                                   "criteria": {"a": "вариант </s> с концом", "b": "обычный"}}}}
    _, candidates, _ = judge.compile(request)
    tok, split = tokenize_rows(candidates, judge.text, judge.config, judge.state_max)
    assert split == ["state", "question q", "option q/a"], split
    for ids in [tok.state, *tok.questions, *(body[:-1] for _, body in tok.options)]:
        assert CONTENT_BREAKERS.isdisjoint(ids)
    rows = build_rows(tok, judge.config, vocab(judge))
    assert all(not check_prompt(r.ids, 0, vocab(judge))[1] for r in rows)


def test_broken_rows_are_refused(text_judge):
    judge = text_judge
    _, rows = rows_of(judge, load_cases()[0]["request"])
    ids = rows[0].ids
    forged = list(ids)
    forged[PREFIX] = forged[PREFIX] + 1 if forged[PREFIX] + 1 not in CONTENT_BREAKERS else 7
    assert check_prompt(forged, 0, vocab(judge))[1] == "state hash does not match the state"
    assert check_prompt(ids[:-1], 0, vocab(judge))[1]              # option without </s>
    assert check_prompt(ids[1:], 0, vocab(judge))[1]               # no leading <pad>
    with pytest.raises(ValueError):
        parse_row([0, 5, 6, 7, 8, 100, 101])


# ----------------------------------------------------------------- small random model
@pytest.fixture(scope="module")
def tiny(model_dir):
    """`Judge` around a small random T5 with FRIDA's tokenizer, and the backend's
    encoder over the same weights."""
    from transformers import T5Config, T5EncoderModel
    from transformers.models.t5.modeling_t5 import T5Attention

    from frida_decisions import Judge
    from frida_decisions.modeling import DecisionEncoder

    v = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    cfg = T5Config(vocab_size=v, d_model=64, d_kv=16, num_heads=4, d_ff=128, num_layers=3,
                   feed_forward_proj="gated-gelu", relative_attention_num_buckets=32,
                   relative_attention_max_distance=128, dropout_rate=0.0,
                   is_encoder_decoder=False, layer_norm_epsilon=1e-6)
    torch.manual_seed(0)
    t5 = T5EncoderModel(cfg).eval()
    with torch.no_grad():
        for name, p in t5.named_parameters():
            if "layer_norm" in name:
                p.uniform_(0.5, 1.5)
        table = next(m for m in t5.modules() if isinstance(m, T5Attention)
                     and m.has_relative_attention_bias)
        table.relative_attention_bias.weight.normal_(0, 2.0)
    head = nn.Linear(64, 1)
    judge = Judge(model_dir, DecisionEncoder(t5, head), "cpu", state_max=384)
    return judge, flat_encoder(t5, head)


def test_tiny_cold_then_cached(tiny, cases):
    """Every case cold (rows of one request share the state within the step), then
    again with new nonces: the second request reads its state from the cache."""
    judge, enc = tiny
    worst = 0.0
    for case in cases:
        ref = torch.tensor(judge.margins_packed(case["request"]))
        candidates, cold = rows_of(judge, case["request"])
        _, warm = rows_of(judge, case["request"])
        paged = Paged(enc, blocks_for(cold, warm))
        state = PREFIX + len(judge.compile(case["request"])[2].state)
        got, used = step(enc, paged, [r.ids for r in cold])
        assert used[0] == 0 and all(h * BS <= state for h in used), used
        d1 = float((ref - margins_of(cold, got, len(candidates))).abs().max())
        got, used = step(enc, paged, [r.ids for r in warm])
        assert all(h * BS > state - BS for h in used) and all(h * BS <= state for h in used), used
        d2 = float((ref - margins_of(warm, got, len(candidates))).abs().max())
        assert max(d1, d2) <= TOL, (case["name"], d1, d2)
        worst = max(worst, d1, d2)
    record("vllm_kernel_vs_judge_tiny", {"cases": len(cases), "max_margin_drift": worst})


def case(name: str) -> dict:
    return next(c for c in load_cases() if c["name"] == name)["request"]


def _two_row_choice(k: int = 20, state: str | None = None) -> dict:
    """A long state and k options: two rows, so the second hits the first within a step."""
    state = state or case("generated/long-state")["state"]
    criteria = {f"t{i}": f"Тема {i}: вариант номер {i}" for i in range(k)}
    return {"state": state, "questions": {"topic": {
        "type": "choice", "instructions": "Какая тема?", "criteria": criteria}}}


@pytest.mark.parametrize("state_len", range(300, 316))
def test_tiny_every_state_length_mod_block(tiny, state_len):
    """Each residue of PREFIX + S mod 16, cold (same-step hits) and repeated."""
    judge, enc = tiny
    request = _two_row_choice()
    saved, judge.state_max = judge.state_max, state_len
    try:
        ref = torch.tensor(judge.margins_packed(request))
        candidates, first = rows_of(judge, request)
        _, second = rows_of(judge, request)
        assert len(first) == 2
        paged = Paged(enc, blocks_for(first, second))
        for rows in (first, second):
            got, used = step(enc, paged, [r.ids for r in rows])
            assert max(used) * BS <= PREFIX + state_len, used
            d = float((ref - margins_of(rows, got, len(candidates))).abs().max())
            assert d <= TOL, d
    finally:
        judge.state_max = saved


def test_tiny_partial_hits_after_eviction(tiny):
    judge, enc = tiny
    request = _two_row_choice(40)
    ref = torch.tensor(judge.margins_packed(request))
    candidates, rows = rows_of(judge, request)
    paged = Paged(enc, 1 + 5 * blocks_for(rows))
    step(enc, paged, [rows[0].ids])                                 # caches the state
    full = paged.longest_hit(rows[1].ids)
    assert full > 2
    for keep in (1, full // 2, full - 1):
        got, _ = step(enc, paged, [r.ids for r in rows], hits=[keep] * len(rows))
        d = float((ref - margins_of(rows, got, len(candidates))).abs().max())
        assert d <= TOL, (keep, d)


def test_tiny_hit_into_a_repeated_nonce_id(tiny):
    """S = 314: PREFIX + S + 1 = 320, so a block ends right after the first nonce id.
    Force every row's first nonce id to repeat: the hit takes it and must stay exact."""
    judge, enc = tiny
    request = _two_row_choice()
    saved, judge.state_max = judge.state_max, 314
    try:
        ref = torch.tensor(judge.margins_packed(request))
        draw = iter(range(10**6))
        candidates, rows = rows_of(judge, request,
                                   nonce=lambda: [777] + [1000 + next(draw) for _ in range(3)])
        got, used = step(enc, Paged(enc, blocks_for(rows)), [r.ids for r in rows])
        assert used == [0, 20], used                                # 320 tokens: the state and N1
        d = float((ref - margins_of(rows, got, len(candidates))).abs().max())
        assert d <= TOL, d
    finally:
        judge.state_max = saved


def test_tiny_other_state_with_the_same_start_does_not_hit(tiny):
    judge, enc = tiny
    text = case("generated/long-state")["state"][:800]
    _, a = rows_of(judge, _two_row_choice(state=text))
    _, b = rows_of(judge, _two_row_choice(state=text + " Совсем другое окончание текста."))
    paged = Paged(enc, blocks_for(a))
    step(enc, paged, [a[0].ids])
    assert a[0].ids[PREFIX:PREFIX + 150] == b[0].ids[PREFIX:PREFIX + 150]
    assert paged.longest_hit(b[0].ids) == 0          # the state hash is in the first block


def test_tiny_resent_rows_are_rejected_not_wrong(tiny):
    """A raw-ids client that resends a row verbatim (same nonce) hits past the state.
    The model must isolate the row and the pooler must return NaN for it."""
    judge, enc = tiny
    _, rows = rows_of(judge, case("generated/long-state"))
    paged = Paged(enc, 2 * blocks_for(rows))
    step(enc, paged, [r.ids for r in rows])
    got, used = step(enc, paged, [r.ids for r in rows], allow_rejected=True)
    assert all(math.isnan(float(m[0])) and m.numel() == 1 for m in got)


@pytest.mark.parametrize("budget,waste", [(1, 1.0), (64 * 1024, 1.25)])
def test_tiny_grouping_does_not_matter(tiny, monkeypatch, budget, waste):
    """The small model fits a whole step in one attention group; force splits down
    to one row per group."""
    judge, enc = tiny
    monkeypatch.setattr(kernel, "GROUP_BYTES", budget)
    monkeypatch.setattr(kernel, "GROUP_WASTE", waste)
    request = case("generated/intent-catalog-40")
    ref = torch.tensor(judge.margins_packed(request))
    candidates, rows = rows_of(judge, request)
    assert len(rows) > 1
    got, _ = step(enc, Paged(enc, blocks_for(rows)), [r.ids for r in rows])
    d = float((ref - margins_of(rows, got, len(candidates))).abs().max())
    assert d <= TOL, d


# ----------------------------------------------------------------- exported model
def test_exported_model_float32(torch_judge, cases):
    """The released weights in float32: cold, then with the state from the cache."""
    judge = torch_judge
    enc = flat_encoder(judge.model.t5, judge.model.head, share=True)
    drift, mismatches = 0.0, []
    for case in cases:
        request = case["request"]
        ref = judge.margins_packed(request)
        candidates, cold = rows_of(judge, request)
        _, warm = rows_of(judge, request)
        paged = Paged(enc, blocks_for(cold, warm))
        parsed = judge.compile(request)[0]
        for rows in (cold, warm):
            got, _ = step(enc, paged, [r.ids for r in rows])
            ours = margins_of(rows, got, len(candidates)).tolist()
            drift = max(drift, max(abs(a - b) for a, b in zip(ref, ours)))
            same = ({q: decision(a) for q, a in aggregate(parsed, candidates, ref).items()} ==
                    {q: decision(a) for q, a in aggregate(parsed, candidates, ours).items()})
            if not same:
                mismatches.append(case["name"])
    record("vllm_kernel_vs_judge_fp32_cpu", {"cases": len(cases), "max_margin_drift": drift,
                                             "decision_mismatches": mismatches})
    assert not mismatches
    assert drift < 1e-4
