"""`FridaDecisionsModel`: FRIDA's encoder as a vLLM pooling model with its own attention.

vLLM flattens the computed tokens of every scheduled prompt into one axis. At the
start of each forward this model reads the request boundaries and cache-hit
lengths from its attention metadata (`attention.FridaAttentionMetadata`), parses
each prompt's ids (`rows.parse_or_isolate`), builds every row's T5 bias and
block mask once, and leaves that plan on the forward context for the 24
attention layers. The layers' arithmetic is `kernel.py`'s, the same code the CPU
parity tests run against `Judge`.

Registered as a *decoder* pooling model: that is the type vLLM gives a paged KV
cache and automatic prefix caching (an `encoder_only` model gets neither). The
attention is not causal -- the mask is ours -- and two vLLM features that
assume causality must therefore be off; the constructor refuses to start
otherwise rather than serve wrong numbers:

* chunked prefill (`--no-enable-chunked-prefill`): a state split across steps
  would have its first chunk computed without seeing the rest;
* `--long-prefill-token-threshold` > 0, which splits prompts even with chunked
  prefill off.

Prefix caching is allowed and is the point of the decoder type: a state cached
by one request is read back by the next one with the same state. With
`--no-enable-prefix-caching` every row carries its state, as in `OnnxJudge`.

Run eager (`--enforce-eager`); nothing here is meant for torch.compile or CUDA
graphs. Padded token rows, if vLLM adds any, are computed in isolation and never
read.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.attention.backend import AttentionType

from .attention import FridaAttentionBackend, FridaAttentionMetadata
from .kernel import BatchPlan, FlatEncoder, dummy_plan, hf_to_local, make_plan, rms_norm
from .pooler import FridaPooler

logger = init_logger(f"vllm.plugins.{__name__}")    # vLLM configures only "vllm.*"

TRACE = os.environ.get("FRIDA_DECISIONS_VLLM_TRACE")          # path: one JSON line per forward


def _wo_dtype(model_dtype: torch.dtype) -> torch.dtype:
    """FFN output projection dtype, as HF loads T5: `_keep_in_fp32_modules = ["wo"]`
    upcasts it for fp16 only (fp16 overflows there); in bf16 and fp32 it is the model's."""
    return torch.float32 if model_dtype == torch.float16 else model_dtype


@attn_type("decoder")
@default_pooling_type(seq_pooling_type="LAST")      # a label for vLLM's config; the pooler is ours
class FridaDecisionsModel(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self._check_config(vllm_config)
        config = vllm_config.model_config.hf_config
        dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.vocab = config.vocab_size
        self.enc = FlatEncoder(config, dtype=dtype, wo_dtype=_wo_dtype(dtype))
        self.attn = nn.ModuleList(
            Attention(num_heads=config.num_heads, head_size=config.d_kv, scale=1.0,
                      num_kv_heads=config.num_heads, cache_config=vllm_config.cache_config,
                      quant_config=None, prefix=maybe_prefix(prefix, f"attn.{i}"),
                      attn_type=AttentionType.DECODER, attn_backend=FridaAttentionBackend)
            for i in range(config.num_layers))
        self.layer_names = [a.layer_name for a in self.attn]
        self.pooler = FridaPooler(self.enc.head, self.vocab)
        self.steps = 0

    @staticmethod
    def _check_config(vllm_config: VllmConfig) -> None:
        sched, cache, par = (vllm_config.scheduler_config, vllm_config.cache_config,
                             vllm_config.parallel_config)
        problems = []
        if sched.enable_chunked_prefill:
            problems.append("chunked prefill is on: pass --no-enable-chunked-prefill")
        if sched.long_prefill_token_threshold:
            problems.append("--long-prefill-token-threshold must be 0")
        if cache.cache_dtype != "auto":
            problems.append(f"--kv-cache-dtype {cache.cache_dtype}: only 'auto' keeps "
                            "cached state keys exact")
        if vllm_config.model_config.attn_type != "decoder":
            problems.append(f"vLLM resolved attn_type {vllm_config.model_config.attn_type!r}, not "
                            "'decoder' (is_causal: false or a CLS pooler in the model config?); "
                            "the attention layers need a paged KV cache")
        if par.tensor_parallel_size != 1 or par.pipeline_parallel_size != 1:
            problems.append("tensor/pipeline parallelism is not implemented (823M, TP=1)")
        if problems:
            raise ValueError("FridaDecisionsModel: " + "; ".join(problems))
        mc = vllm_config.model_config
        logger.info("FridaDecisionsModel: attn_type=%s prefix_caching=%s chunked_prefill=%s "
                    "enforce_eager=%s dtype=%s wo_dtype=%s max_model_len=%d",
                    mc.attn_type, cache.enable_prefix_caching, sched.enable_chunked_prefill,
                    mc.enforce_eager, mc.dtype, _wo_dtype(mc.dtype), mc.max_model_len)
        if not mc.enforce_eager:
            logger.warning("FridaDecisionsModel is meant to run with --enforce-eager")

    # ----------------------------------------------------------------- plan
    def _metadata(self) -> FridaAttentionMetadata | None:
        md = get_forward_context().attn_metadata
        if isinstance(md, list):
            md = md[0] if md else None
        if isinstance(md, dict):
            md = md.get(self.layer_names[0])
        return md if isinstance(md, FridaAttentionMetadata) else None

    def _plan(self, input_ids: torch.Tensor, positions: torch.Tensor) -> BatchPlan:
        device = input_ids.device
        meta = self._metadata()
        if meta is None:
            # Profiling and warm-up runs carry no metadata. Rows of max_model_len
            # with full attention make vLLM's memory profile see the largest bias
            # tensors a real batch can have.
            return dummy_plan(self.enc, input_ids.shape[0], self.max_model_len, device)
        total = meta.num_actual_tokens
        qsl = meta.query_start_loc_cpu.tolist()
        seq = meta.seq_lens_cpu.tolist()
        plan = make_plan(self.enc, input_ids[:total].tolist(), qsl, seq, meta.block_table,
                         meta.block_size, device)
        # vLLM's positions must count from the prompt start: the first computed
        # token of a row sits at its number of cached tokens. Checked every step --
        # the layout above depends on it, and a runner change would break it silently.
        if total:
            first = positions[torch.tensor(qsl[:-1], device=device, dtype=torch.long)].tolist()
            for i, row in enumerate(plan.rows):
                expect = seq[i] - row.n
                if row.n and first[i] != expect:
                    raise RuntimeError(
                        f"vLLM positions start at {first[i]} for a row with {expect} cached "
                        "tokens; FridaDecisionsModel's layout assumes positions count from the "
                        "prompt start")
        if TRACE:
            with open(TRACE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "t": time.time(), "step": self.steps, "tokens": total,
                    "padded": int(input_ids.shape[0]),
                    "rows": [{"n": r.n, "cached": seq[i] - r.n, "read": r.cached,
                              "valid": r.parsed.valid, "state_len": r.parsed.state_len,
                              "options": len(r.parsed.options), "error": r.parsed.error}
                             for i, r in enumerate(plan.rows)]}) + "\n")
        return plan

    # -------------------------------------------------------------- forward
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.enc.embed(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors=None, inputs_embeds: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        ctx = get_forward_context()
        ctx.frida_batch = self._plan(input_ids, positions)
        self.steps += 1
        try:
            x = self.enc.embed(input_ids) if inputs_embeds is None else inputs_embeds
            t = x.shape[0]
            for block, attn in zip(self.enc.blocks, self.attn):
                q, k, v = block.qkv(x)
                out = attn(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
                x = block.finish(x, out)
            return rms_norm(x, self.enc.final_ln, self.enc.eps)
        finally:
            ctx.frida_batch = None

    # -------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped = 0
        for name, tensor in weights:
            local = hf_to_local(name)
            if local is None:
                skipped += 1
                continue
            full = f"enc.{local}"
            param = params[full]
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            loaded.add(full)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"FridaDecisionsModel: weights missing from the checkpoint: "
                             f"{sorted(missing)[:6]}{' ...' if len(missing) > 6 else ''} "
                             "(the head comes from head.safetensors next to the model)")
        logger.info("FridaDecisionsModel: loaded %d tensors, skipped %d", len(loaded), skipped)
        return loaded
