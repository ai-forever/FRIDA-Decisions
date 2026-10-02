"""FRIDA's attention as a vLLM attention backend.

vLLM's own backends know causal and sliding-window masks and no additive bias.
This one computes exactly `kernel.attend`: per row, queries of the tokens computed
in this step against `[keys read from the paged KV cache ; keys computed now]`,
with the row's T5 bias and block mask.

What the backend cannot know from vLLM's metadata -- which token is state,
question, option or marker, and the bias that follows -- the model works out
from the ids at the start of its forward and leaves on the forward context as
`frida_batch` (`kernel.BatchPlan`). Everything there is plain eager Python; the
model is not compiled.

KV cache: vLLM writes every computed token's K/V through `do_kv_cache_update`
before `forward` is called for the layer, as for any backend with
`forward_includes_kv_cache_update = False`. Only state keys are ever read back
(the nonce right after the state keeps prefix-cache hits inside it); writing
the rest costs a little bandwidth and keeps the write path vLLM's.
Layout: vLLM's logical per-layer `[blocks, heads, block_size, 2 * head_size]`,
K in the first half of the last axis, V in the second.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionImpl,
                                       AttentionMetadataBuilder, AttentionType,
                                       CommonAttentionMetadata)

from .kernel import kv_write, run_attention


@dataclass
class FridaAttentionMetadata:
    num_actual_tokens: int
    query_start_loc_cpu: torch.Tensor      # (R+1,)
    seq_lens_cpu: torch.Tensor             # (R,) cached + computed, per request
    block_table: torch.Tensor              # (R, max_blocks) on device
    slot_mapping: torch.Tensor             # (T,) on device, -1 for padding
    block_size: int


class FridaAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["FridaAttentionImpl"]:
        return FridaAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FridaMetadataBuilder"]:
        return FridaMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []


class FridaMetadataBuilder(AttentionMetadataBuilder[FridaAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FridaAttentionMetadata:
        cm = common_attn_metadata
        seq = cm.seq_lens_cpu_upper_bound           # exact for prefill rows, all ours are
        if seq is None:
            seq = cm.seq_lens.cpu()
        return FridaAttentionMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc_cpu=cm.query_start_loc_cpu,
            seq_lens_cpu=seq,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            block_size=self.kv_cache_spec.block_size)


class FridaAttentionImpl(AttentionImpl):
    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None, alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype: str = "auto", logits_soft_cap=None,
                 attn_type: str = AttentionType.DECODER,
                 kv_sharing_target_layer_name: str | None = None, **kwargs) -> None:
        self.num_heads, self.head_size, self.scale = num_heads, head_size, scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        unsupported = {
            "num_kv_heads != num_heads": self.num_kv_heads != num_heads,
            "alibi": alibi_slopes is not None,
            "sliding window": sliding_window is not None,
            "logits soft cap": logits_soft_cap is not None,
            "KV sharing": kv_sharing_target_layer_name is not None,
            f"attn_type {attn_type}": attn_type != AttentionType.DECODER,
            # Reusing cached state keys is exact only if they are the bytes the
            # first request computed; a quantized cache would round them.
            f"kv_cache_dtype {kv_cache_dtype}": kv_cache_dtype != "auto",
        }
        bad = [name for name, hit in unsupported.items() if hit]
        if bad:
            raise NotImplementedError(f"FRIDA attention does not support: {', '.join(bad)}")

    def do_kv_cache_update(self, layer, key: torch.Tensor, value: torch.Tensor,
                           kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        if kv_cache.numel():
            kv_write(kv_cache, key, value, slot_mapping)

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                kv_cache: torch.Tensor, attn_metadata, output: torch.Tensor,
                output_scale=None, output_block_scale=None) -> torch.Tensor:
        plan = getattr(get_forward_context(), "frida_batch", None)
        if plan is None:
            raise RuntimeError("FRIDA attention called without a row plan: "
                               "the layer was used outside FridaDecisionsModel.forward")
        return run_attention(plan, query, key, value, kv_cache, output)
