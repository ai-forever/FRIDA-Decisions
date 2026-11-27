"""The arithmetic of the vLLM backend's forward, with no vLLM in it.

Everything numerical the backend does lives here, so the parity tests can run it
on CPU against `Judge` (`modeling.DecisionEncoder`) without a vLLM install, and
the vLLM model (`model.py`) only decides where the rows and the cached keys come
from.

The attention kernel is written for the general case: the keys of a row are
`[cached state keys ; keys of the tokens computed now]`. Without a prefix cache
(or on a miss) the cached part is empty. With one, it holds the state tokens an
earlier request already computed, at positions `0..cached-1`, and the mask and
the relative bias treat them exactly as if they had been computed in this step:
state keys are visible to every non-marker token, and nothing about a state
token depends on what follows the state.

Numerics follow HF `T5EncoderModel` as `Judge` runs it, not an idealised T5: RMS
norm with the variance in fp32 and the result cast to the weight dtype;
attention without the 1/sqrt(d) scale; the relative bias and the mask in the
model dtype, added inside SDPA. The FFN output projection `wo` runs in fp32 when
the model runs in fp16 (as HF's `_keep_in_fp32_modules = ["wo"]` loads it; fp16
overflows there); in bf16 and fp32 it is in the model dtype, as in `Judge`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..packing import relative_buckets
from .rows import MARK, STATE, Parsed, check_prompt, isolated_state, parse_or_isolate

ALIGN = 8          # key width of the bias buffer is padded to this; see `row_bias`


# ------------------------------------------------------------- mask and bias
_LUTS: dict = {}


def buckets(q_pos: torch.Tensor, k_pos: torch.Tensor, num_buckets: int,
            max_distance: int) -> torch.Tensor:
    """T5's bidirectional relative-position bucket, query positions down, keys across.

    The table is `packing.relative_buckets`, the one `Judge` and `OnnxJudge` use,
    built once on the CPU (a GPU `log` may round the bucket edges at distances
    16, 32 and 64 differently) and only indexed on the device. Every distance at
    or past `max_distance` falls in the last bucket of its sign, so the table
    covers [-max_distance, max_distance] and the distance is clamped into it."""
    key = (num_buckets, max_distance, q_pos.device)
    lut = _LUTS.get(key)
    if lut is None:
        rel = np.arange(-max_distance, max_distance + 1, dtype=np.int64)
        table = relative_buckets(np.zeros(1, dtype=np.int64), rel, num_buckets, max_distance)[0]
        lut = _LUTS[key] = torch.from_numpy(table.astype(np.int64)).to(q_pos.device)
    rel = (k_pos[None, :] - q_pos[:, None]).clamp(-max_distance, max_distance)
    return lut[rel + max_distance]


class RowTensors:
    """One row's layout as tensors on the model device.

    `cached` is how many keys come from the KV cache before the computed tokens:
    the parsed cache hit, or 0 for a row that did not parse (its tokens attend
    to themselves only, so no cached key is read)."""

    def __init__(self, parsed: Parsed, device):
        self.valid = parsed.valid
        self.cached = parsed.cached if parsed.valid else 0
        self.cached_marks = parsed.cached_marks if parsed.valid else 0
        self.cached_tail = parsed.cached_tail if parsed.valid else 0
        self.n = len(parsed.kind)
        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
        self.kind, self.seg, self.parent, self.pos = (
            t(parsed.kind), t(parsed.seg), t(parsed.parent), t(parsed.pos))
        self.options = parsed.options

    def cached_keys(self):
        """kind, segment and position of the cached keys: `<pad> H{4}` markers, the
        state, and nonce markers if the hit reached into the nonce."""
        c, m, tail, device = self.cached, self.cached_marks, self.cached_tail, self.kind.device
        state = c - m - tail
        marks = lambda k, base: (torch.full((k,), MARK, device=device),
                                 base - torch.arange(k, device=device),       # unique, matches nothing
                                 torch.zeros(k, dtype=torch.long, device=device))
        head, tail_ = marks(m, -2), marks(tail, -2 - m)
        kind = torch.cat([head[0], torch.full((state,), STATE, device=device), tail_[0]])
        seg = torch.cat([head[1], torch.zeros(state, dtype=torch.long, device=device), tail_[1]])
        pos = torch.cat([head[2], torch.arange(state, device=device), tail_[2]])
        return kind, seg, pos


def allowed(row: RowTensors) -> torch.Tensor:
    """(n, cached + n) bool: may computed token i attend to key j.

    Keys are `[cached ; computed]`. Rules, as in `packing.pack_rows`: a token sees
    its own block; every non-marker sees the state; an option also sees its
    question. Markers are blocks of one token that nobody else sees.
    """
    c_kind, c_seg, _ = row.cached_keys()
    k_kind = torch.cat([c_kind, row.kind])
    k_seg = torch.cat([c_seg, row.seg])
    same = row.seg[:, None] == k_seg[None, :]
    state = (k_kind == STATE)[None, :] & (row.kind != MARK)[:, None]
    parent = row.parent[:, None] == k_seg[None, :]
    return same | state | parent


def row_bias(row: RowTensors, table: torch.Tensor, num_buckets: int,
             max_distance: int, dtype) -> torch.Tensor:
    """(1, H, n, cached + n) additive bias: T5 relative bias, `finfo.min` where masked.

    Built once per row and shared by all 24 layers, as T5 shares block 0's bias.
    The key dimension lives in a buffer padded to a multiple of ALIGN and is
    returned as a view: the memory-efficient SDPA kernel wants an aligned row
    stride and would otherwise pad a copy of the bias in every layer.
    """
    c, n, device = row.cached, row.n, row.kind.device
    k_pos = torch.cat([row.cached_keys()[2], row.pos])
    b = buckets(row.pos, k_pos, num_buckets, max_distance)
    width = -(-(c + n) // ALIGN) * ALIGN
    out = torch.empty((1, table.shape[1], n, width), dtype=dtype, device=device)
    view = out[..., :c + n]
    view.copy_(table.to(dtype)[b].permute(2, 0, 1)[None])
    view.masked_fill_(~allowed(row)[None, None], torch.finfo(dtype).min)
    return view


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           bias: torch.Tensor) -> torch.Tensor:
    """q (n, H, d), k/v (cached + n, H, d), bias (1, H, n, cached + n) -> (n, H*d)."""
    n, h, d = q.shape
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
        attn_mask=bias, scale=1.0)
    return out[0].transpose(0, 1).reshape(n, h * d)


# ------------------------------------------------------------ paged KV cache
# vLLM's per-layer cache, logical [blocks, heads, block_size, 2 * head_dim]: K in
# the first half of the last axis, V in the second. The physical order behind the
# view is vLLM's business (by default [blocks, block_size, heads, C]); indexing
# through the logical view is correct for any of them.
def kv_write(kv_cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    """Store K/V (T, H, D) at flat slots `block * block_size + offset`.

    Padding slots are -1 and go to the null block 0, which vLLM never hands to a
    request (the same trick as vLLM's `basic_cache`)."""
    bs = kv_cache.shape[2]
    slots = slot_mapping[:key.shape[0]].clamp_min(0)
    kv_cache[slots // bs, :, slots % bs] = torch.cat([key, value], dim=-1)


# Rows of one step are attended in padded groups, one SDPA per group and layer:
# a 50-row request is then ~50x fewer kernel launches than one call per row. A
# group's working set -- its bias (R, H, Nq, Nk) and the K/V gathered for it
# (R, Nk, H, 2D) -- stays under GROUP_BYTES, and padding under GROUP_WASTE of
# the real (query, key) area.
GROUP_BYTES = 128 * 2**20
GROUP_WASTE = 1.25


@dataclass
class RowPlan:
    start: int                       # first computed token of the row in the flat batch
    n: int                           # computed tokens
    cached: int                      # keys before them that the row attends over
    key_from: int                    # prompt position of the row's first key
    bias: torch.Tensor | None        # (1, H, n, cached + n); moved into its group
    parsed: Parsed


@dataclass
class GroupPlan:
    rows: list[int]
    q_index: torch.Tensor            # (R, Nq) flat index of each query; padding -> 0
    k_blocks: torch.Tensor           # (R, Nk) KV-cache block of each key; padding -> its first key
    k_offsets: torch.Tensor          # (R, Nk) slot in the block
    bias: torch.Tensor               # (R, H, Nq, Nk), finfo.min on padding
    out_index: torch.Tensor          # (M,) flat token index of every real query
    out_src: torch.Tensor            # (M,) its place in the flattened (R * Nq) result


@dataclass
class BatchPlan:
    rows: list[RowPlan]
    num_tokens: int                  # real tokens; anything after them is padding
    block_size: int
    groups: list[GroupPlan] | None = None   # None: per row, computed keys only (profiling)
    reserve_bytes: int = 0                  # profiling: stand-in for a group's working set


def _groups(rows: list[RowPlan], heads: int, head_dim: int, itemsize: int) -> list[list[int]]:
    order = sorted(range(len(rows)), key=lambda i: (-rows[i].n, -(rows[i].cached + rows[i].n)))
    out, cur = [], []
    for i in order:
        cand = cur + [i]
        nq = max(rows[j].n for j in cand)
        nk = -(-max(rows[j].cached + rows[j].n for j in cand) // ALIGN) * ALIGN
        size = len(cand) * nk * heads * (nq + 2 * head_dim) * itemsize
        real = sum(rows[j].n * (rows[j].cached + rows[j].n) for j in cand)
        if cur and (size > GROUP_BYTES or len(cand) * nq * nk > GROUP_WASTE * real):
            out.append(cur)
            cur = [i]
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def make_plan(enc: "FlatEncoder", ids: list[int], query_start: list[int], seq_lens: list[int],
              block_table: torch.Tensor, block_size: int, device) -> BatchPlan:
    """Parse every row of a step and build what attention needs, once for all layers.

    `ids` are the computed tokens of the step, flat; row i is
    `ids[query_start[i]:query_start[i+1]]` and has `seq_lens[i] - its length`
    tokens before it in the KV cache. Every key -- cached or computed now -- is
    read back from the paged cache (vLLM writes the step's K/V before attention),
    so a row's keys are one gather whatever mix of hit and miss it is."""
    rows = []
    for i in range(len(query_start) - 1):
        start, end = query_start[i], query_start[i + 1]
        n, in_cache = end - start, seq_lens[i] - (end - start)
        parsed = parse_or_isolate(ids[start:end], in_cache)
        rt = RowTensors(parsed, device)
        rows.append(RowPlan(start, n, rt.cached, in_cache - rt.cached, enc.bias(rt), parsed))
    heads, d = enc.n_heads, enc.d_kv
    dtype = enc.embed.weight.dtype
    floor = torch.finfo(dtype).min
    itemsize = torch.empty((), dtype=dtype).element_size()
    groups = []
    for members in _groups(rows, heads, d, itemsize):
        r = len(members)
        nq = max(rows[i].n for i in members)
        nk = -(-max(rows[i].cached + rows[i].n for i in members) // ALIGN) * ALIGN
        q_index = torch.zeros((r, nq), dtype=torch.long, device=device)
        k_blocks = torch.zeros((r, nk), dtype=torch.long, device=device)
        k_offsets = torch.zeros((r, nk), dtype=torch.long, device=device)
        bias = torch.full((r, heads, nq, nk), floor, dtype=dtype, device=device)
        out_index, out_src = [], []
        for slot, i in enumerate(members):
            row = rows[i]
            keys = row.cached + row.n
            q_index[slot, :row.n] = torch.arange(row.start, row.start + row.n, device=device)
            pos = torch.arange(row.key_from, row.key_from + keys, device=device)
            blocks = block_table[i].long()[pos // block_size]
            # Padding keys point at the row's own first key: masked out, but read all
            # the same, and a key from the null block can hold anything, NaN
            # included -- q.NaN + -inf and 0 * NaN are both NaN.
            k_blocks[slot] = blocks[0]
            k_offsets[slot] = pos[0] % block_size
            k_blocks[slot, :keys] = blocks
            k_offsets[slot, :keys] = pos % block_size
            bias[slot, :, :row.n, :keys] = row.bias[0]
            bias[slot, :, row.n:, 0] = 0          # padded queries: one visible key, no NaN
            row.bias = None
            out_index.append(torch.arange(row.start, row.start + row.n, device=device))
            out_src.append(torch.arange(slot * nq, slot * nq + row.n, device=device))
        groups.append(GroupPlan(members, q_index, k_blocks, k_offsets, bias,
                                torch.cat(out_index), torch.cat(out_src)))
    return BatchPlan(rows, query_start[-1], block_size, groups)


def dummy_plan(enc: "FlatEncoder", total: int, row_len: int, device) -> BatchPlan:
    """Rows of `row_len` with full attention: the largest bias a real batch can need,
    plus one group's working set held during attention, so that vLLM's memory
    profile accounts for both."""
    rows = []
    for start in range(0, total, row_len):
        n = min(row_len, total - start)
        parsed = isolated_state(n)
        rows.append(RowPlan(start, n, 0, 0, enc.bias(RowTensors(parsed, device)), parsed))
    return BatchPlan(rows, total, 0, None, GROUP_BYTES)


def run_attention(plan: BatchPlan, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  kv_cache: torch.Tensor | None, output: torch.Tensor) -> torch.Tensor:
    """One layer: every row's queries against its keys -- cached and computed --
    with its bias and mask, group by group.

    query/key/value/output are (T, H, D). The step's K/V must already be in
    `kv_cache` (vLLM writes them before calling the backend), which is also what
    makes a hit on blocks another row of the same step is computing right now
    correct."""
    d = query.shape[-1]
    if plan.num_tokens < output.shape[0]:
        output[plan.num_tokens:].zero_()
    if plan.groups is None:                       # profiling: no cache, computed keys only
        held = torch.empty(plan.reserve_bytes, dtype=torch.uint8, device=query.device)
        for row in plan.rows:
            sl = slice(row.start, row.start + row.n)
            output[sl] = attend(query[sl], key[sl], value[sl], row.bias).view(row.n, -1, d)
        assert held.numel() == plan.reserve_bytes
        return output
    if kv_cache.shape[2] != plan.block_size:
        raise RuntimeError(f"KV cache block size {kv_cache.shape[2]} != "
                           f"metadata block size {plan.block_size}")
    cache = kv_cache.transpose(1, 2)                                  # [blocks, block_size, H, 2D]
    for g in plan.groups:
        r, nq = g.q_index.shape
        q = query[g.q_index].transpose(1, 2)                          # (R, H, Nq, D)
        kv = cache[g.k_blocks, g.k_offsets]                           # (R, Nk, H, 2D)
        k, v = kv[..., :d].transpose(1, 2), kv[..., d:].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=g.bias, scale=1.0)
        output[g.out_index] = a.transpose(1, 2).reshape(r * nq, -1, d)[g.out_src]
    return output


def readout_plan(prompts: list[list[int]], scheduled: list[int], first: list[int], vocab: int):
    """Where each prompt's option tokens sit among the hidden states of this step.

    prompts: every prompt's full ids; scheduled: tokens computed this step;
    first: index of the prompt's first computed token in the flat hidden states.
    The computed part is parsed with the model's own parser (`check_prompt`), so
    a row the model computed in isolation can never be read out as an answer.
    Returns (index, slot, counts, per_prompt, errors): token indices, the option
    each belongs to, tokens per option, options per prompt (-1: rejected), and the
    rejection reasons by prompt."""
    index, slot, counts, per_prompt, errors = [], [], [], [], {}
    for i, ids in enumerate(prompts):
        part, error = check_prompt(ids, len(ids) - scheduled[i], vocab)
        if error:
            errors[i] = error
            per_prompt.append(-1)
            continue
        per_prompt.append(len(part.options))
        for start, end in part.options:
            index.extend(range(first[i] + start, first[i] + end))
            slot.extend([len(counts)] * (end - start))
            counts.append(end - start)
    return index, slot, counts, per_prompt, errors


def pool_margins(hidden: torch.Tensor, head: nn.Linear, plan) -> list[torch.Tensor]:
    """Span means in fp32 through the head; NaN for a rejected prompt."""
    index, slot, counts, per_prompt, _ = plan
    device = hidden.device
    margins = None
    if counts:
        picked = hidden[torch.tensor(index, device=device)].float()
        pooled = torch.zeros((len(counts), picked.shape[-1]), device=device)
        pooled.index_add_(0, torch.tensor(slot, device=device), picked)
        pooled /= torch.tensor(counts, device=device, dtype=torch.float32)[:, None]
        margins = head(pooled).squeeze(-1)
    out, k = [], 0
    for n in per_prompt:
        if n < 0:
            out.append(torch.full((1,), float("nan"), device=device))
        else:
            out.append(margins[k:k + n])
            k += n
    return out


# --------------------------------------------------------------------- layers
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`T5LayerNorm.forward`, operation for operation."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        x = x.to(weight.dtype)
    return weight * x


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5Block(nn.Module):
    """One encoder block. Parameter names are HF's, so weights load by name."""

    def __init__(self, d_model: int, n_heads: int, d_kv: int, d_ff: int, eps: float, dtype,
                 wo_dtype=torch.float32):
        super().__init__()
        inner = n_heads * d_kv
        self.n_heads, self.d_kv, self.eps = n_heads, d_kv, eps
        lin = lambda i, o, dt=dtype: nn.Linear(i, o, bias=False, dtype=dt)
        self.q, self.k, self.v, self.o = lin(d_model, inner), lin(d_model, inner), lin(d_model, inner), lin(inner, d_model)
        self.ln_attn = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.wi_0, self.wi_1 = lin(d_model, d_ff), lin(d_model, d_ff)
        self.wo = lin(d_ff, d_model, wo_dtype)
        self.ln_ff = nn.Parameter(torch.ones(d_model, dtype=dtype))

    def qkv(self, x):
        h = rms_norm(x, self.ln_attn, self.eps)
        shape = (x.shape[0], self.n_heads, self.d_kv)
        return self.q(h).view(shape), self.k(h).view(shape), self.v(h).view(shape)

    def finish(self, x, attn_out):
        """Residual after attention, then the FFN with its residual."""
        x = x + self.o(attn_out)
        h = rms_norm(x, self.ln_ff, self.eps)
        h = gelu_new(self.wi_0(h)) * self.wi_1(h)
        if h.dtype != self.wo.weight.dtype:
            h = h.to(self.wo.weight.dtype)
        return x + self.wo(h)


HF_NAMES = {
    "layer.0.SelfAttention.q.weight": "q.weight",
    "layer.0.SelfAttention.k.weight": "k.weight",
    "layer.0.SelfAttention.v.weight": "v.weight",
    "layer.0.SelfAttention.o.weight": "o.weight",
    "layer.0.layer_norm.weight": "ln_attn",
    "layer.1.DenseReluDense.wi_0.weight": "wi_0.weight",
    "layer.1.DenseReluDense.wi_1.weight": "wi_1.weight",
    "layer.1.DenseReluDense.wo.weight": "wo.weight",
    "layer.1.layer_norm.weight": "ln_ff",
}


def hf_to_local(name: str) -> str | None:
    """`encoder.block.3.layer.0.SelfAttention.q.weight` -> `blocks.3.q.weight`, etc.

    Returns None for weights the model does not use (the decoder of a full T5,
    LM head). The relative bias table lives in block 0 in HF and is pulled out.
    `head.safetensors` stores the head as bare `weight` and `bias`."""
    for prefix in ("encoder.", ""):
        if name.startswith(prefix + "block."):
            rest = name[len(prefix + "block."):]
            idx, _, tail = rest.partition(".")
            if tail == "layer.0.SelfAttention.relative_attention_bias.weight":
                return "rel_bias.weight" if idx == "0" else None
            local = HF_NAMES.get(tail)
            return f"blocks.{idx}.{local}" if local else None
    if name in ("shared.weight", "encoder.embed_tokens.weight", "embed_tokens.weight"):
        return "embed.weight"
    if name in ("encoder.final_layer_norm.weight", "final_layer_norm.weight"):
        return "final_ln"
    if name in ("head.weight", "head.bias"):
        return name
    if name in ("weight", "bias"):
        return f"head.{name}"
    return None


class FlatEncoder(nn.Module):
    """FRIDA's encoder plus the decision head, over a flat batch of rows.

    `forward_rows` is the whole computation for rows whose cached keys (if any)
    are given as per-layer tensors -- what the CPU tests use. The vLLM model
    calls the same pieces (`blocks[i].qkv`, `attend`, `blocks[i].finish`) with
    keys from vLLM's paged cache instead.
    """

    def __init__(self, config, dtype=torch.float32, wo_dtype=torch.float32):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = getattr(config, "relative_attention_max_distance", 128)
        self.n_heads, self.d_kv = config.num_heads, config.d_kv
        self.eps = config.layer_norm_epsilon
        self.dtype = dtype
        self.embed = nn.Embedding(config.vocab_size, config.d_model, dtype=dtype)
        self.rel_bias = nn.Embedding(self.num_buckets, config.num_heads, dtype=dtype)
        self.blocks = nn.ModuleList(
            T5Block(config.d_model, config.num_heads, config.d_kv, config.d_ff,
                    self.eps, dtype, wo_dtype) for _ in range(config.num_layers))
        self.final_ln = nn.Parameter(torch.ones(config.d_model, dtype=dtype))
        self.head = nn.Linear(config.d_model, 1, dtype=torch.float32)

    def load_hf(self, weights, head=None) -> set[str]:
        """Load from (name, tensor) pairs in HF T5 naming. Returns the local names loaded."""
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            local = hf_to_local(name)
            if local is None:
                continue
            with torch.no_grad():
                params[local].copy_(tensor.to(params[local].dtype))
            loaded.add(local)
        if head is not None:
            for name, tensor in head.items():
                with torch.no_grad():
                    params[f"head.{name}"].copy_(tensor.float())
                loaded.add(f"head.{name}")
        return loaded

    def bias(self, row: RowTensors) -> torch.Tensor:
        return row_bias(row, self.rel_bias.weight, self.num_buckets, self.max_distance,
                        self.embed.weight.dtype)

    def forward_rows(self, ids: torch.Tensor, rows: list[tuple[int, RowTensors, list | None]],
                     capture: list | None = None):
        """ids (T,) flat; rows: (start index in ids, layout, per-layer cached (k, v) or None).

        Returns the final hidden states (T, d_model) in the model dtype. With
        `capture`, appends each layer's (k, v) of all T tokens -- what vLLM writes
        to its KV cache in that step (tests use it as the cache)."""
        x = self.embed(ids)
        biases = [self.bias(r) for _, r, _ in rows]
        for i, block in enumerate(self.blocks):
            q, k, v = block.qkv(x)
            if capture is not None:
                capture.append((k, v))
            out = torch.empty((x.shape[0], self.n_heads * self.d_kv), dtype=q.dtype, device=q.device)
            for (start, row, cache), bias in zip(rows, biases):
                sl = slice(start, start + row.n)
                kk, vv = k[sl], v[sl]
                if row.cached:
                    ck, cv = cache[i]
                    kk, vv = torch.cat([ck[:row.cached], kk]), torch.cat([cv[:row.cached], vv])
                out[sl] = attend(q[sl], kk, vv, bias)
            x = block.finish(x, out)
        return rms_norm(x, self.final_ln, self.eps)

    def readout(self, hidden: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        """Mean over each option's tokens in fp32, then the head: one margin per span."""
        pooled = torch.stack([hidden[s:e].float().mean(0) for s, e in spans])
        return self.head(pooled).squeeze(-1)
