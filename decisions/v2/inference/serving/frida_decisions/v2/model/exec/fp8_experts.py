"""FP8 experts for a MoE block under `CausalFlatExecutor`.

At one request the 35B reads nearly all its experts' weights in every layer (256 experts,
8 a token), so the grouped matmul is bound by memory: about 1.6 GB a layer in bf16.

Block FP8 (`block_fp8_weights`, `vllm_experts`), the executor `flat_fp8`: the format of
Qwen's own `-FP8` repo and of our export (`model/vllm_fp8.py`): e4m3 in 128x128 blocks with
a bf16 scale, the tokens quantized in groups of 128 inside vLLM's fused MoE kernel. The
numbers are the export's, so the plugin serves what the `-FP8` artifact holds. It needs
vLLM, so it runs where the plugin runs.

Rowwise (`fp8_weights`, `grouped_mm_experts_fp8`): one scale per expert and output row, tokens
quantized per row, `torch._scaled_grouped_mm`. Each expert's rows start at a multiple of
16, which that kernel wants; the pad rows are zero and nothing reads them. Everything else
is `grouped_mm_experts_forward` of transformers: the sort by expert, the routing weights,
the order restored and the sum over the top k in float32.
"""
from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
ALIGN = 16
BLOCK = 128


def encode_blocks(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, R, C) as e4m3 and a float32 scale per 128x128 block: `model/vllm_fp8.encode_block`
    per expert, the same rounding (the fp32 scale divides, the stored one is rounded to bf16)."""
    experts, rows, cols = w.shape
    if rows % BLOCK or cols % BLOCK:
        raise ValueError(f"{rows}x{cols} is not whole {BLOCK}x{BLOCK} blocks")
    x = w.float().view(experts, rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    amax = x.abs().amax(dim=(2, 4), keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(experts, rows, cols), scale.to(torch.bfloat16).float().view(experts, rows // BLOCK, cols // BLOCK)


def block_fp8_weights(experts) -> dict:
    """The experts' weights in block FP8, on the module itself: `gate_up_proj` and `down_proj`
    become e4m3, each with a `<name>_scale` buffer. So the module's tensors are what a serving
    folder stores (`vllm_hybrid/weights.py`), and a module read back from one, already in FP8,
    is taken as it is. A layer at a time keeps the peak low: the bf16 weights go, which is the
    point on a card the 35B nearly fills."""
    out = {}
    for name in ("gate_up_proj", "down_proj"):
        param = getattr(experts, name)
        if param.dtype != FP8:
            q = torch.empty(param.shape, dtype=FP8, device=param.device)
            scales = []
            for e in range(0, param.shape[0], 32):
                q[e:e + 32], s = encode_blocks(param.data[e:e + 32])
                scales.append(s)
            param.data = q
            experts.register_buffer(name + "_scale", torch.cat(scales))
        out[name], out[name + "_scale"] = param.data, getattr(experts, name + "_scale")
    return out


_deep_gemm_probed = False


def probe_deep_gemm_apart() -> None:
    """vLLM's first FP8 quantization of the tokens probes DeepGEMM (`is_deep_gemm_e8m0_used` ->
    `_lazy_init`). Where DeepGEMM's JIT finds no CUDA toolkit, its `set_pdl`
    raises, and vLLM hands the exception to `logger.warning_once`, an lru_cache that keeps its
    arguments for good: the exception, its traceback and every frame up to the caller, whose
    locals hold the model. Run here
    once in a thread of its own, the same calls in the same order, so the E8M0 choice is as
    before and what is kept is that thread's frames."""
    global _deep_gemm_probed
    if _deep_gemm_probed:
        return
    import threading

    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    thread = threading.Thread(target=is_deep_gemm_e8m0_used, name="vllm-deep-gemm-probe")
    thread.start()
    thread.join()
    _deep_gemm_probed = True


def vllm_experts(weights: dict, hidden: torch.Tensor, top_k_index: torch.Tensor,
                 top_k_weights: torch.Tensor) -> torch.Tensor:
    """vLLM's fused MoE on block FP8 weights; gate then up in `gate_up_proj`, as both read it."""
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    probe_deep_gemm_apart()
    config = fp8_w8a8_moe_quant_config(w1_scale=weights["gate_up_proj_scale"], w2_scale=weights["down_proj_scale"],
                                       block_shape=[BLOCK, BLOCK])
    return fused_experts(hidden, weights["gate_up_proj"], weights["down_proj"], top_k_weights.float(),
                         top_k_index.to(torch.int32), quant_config=config)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`x` (..., K) as float8 with one float32 scale per row."""
    scale = x.abs().amax(-1).float().clamp(min=1e-12) / FP8_MAX
    return (x.float() / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def fp8_weights(experts) -> dict:
    """The experts' weights as float8, the layout `_scaled_grouped_mm` reads: (E, K, N) column-major."""
    gate_up, s_gate_up = quantize_rows(experts.gate_up_proj.data)      # (E, 2I, D), (E, 2I)
    down, s_down = quantize_rows(experts.down_proj.data)               # (E, D, I), (E, D)
    return {"gate_up": gate_up.transpose(-2, -1), "s_gate_up": s_gate_up,
            "down": down.transpose(-2, -1), "s_down": s_down}


def _aligned_rows(ids_sorted: torch.Tensor, experts: int):
    """Row of each sorted pair in a layout where each expert's rows start at a multiple of
    ALIGN, and the cumulative ends of the groups. Static size: no sync."""
    counts = torch.histc(ids_sorted.int(), bins=experts, min=0, max=experts - 1).long()
    padded = (counts + ALIGN - 1) // ALIGN * ALIGN
    starts = torch.cumsum(padded, 0) - padded
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(ids_sorted.numel(), device=ids_sorted.device) - first[ids_sorted]
    rows = starts[ids_sorted] + rank
    return rows, torch.cumsum(padded, 0).to(torch.int32)


def grouped_mm_experts_fp8(experts, weights: dict, hidden: torch.Tensor, top_k_index: torch.Tensor,
                           top_k_weights: torch.Tensor) -> torch.Tensor:
    tokens, k = top_k_index.shape
    size = hidden.size(-1)
    total = experts.num_experts
    sample_weights = top_k_weights.reshape(-1)
    ids_sorted, perm = torch.sort(top_k_index.reshape(-1))
    rows, offs = _aligned_rows(ids_sorted, total)
    capacity = ids_sorted.numel() + total * (ALIGN - 1)
    inputs = hidden.new_zeros(capacity, size)
    inputs.index_copy_(0, rows, hidden[perm // k])
    a, s_a = quantize_rows(inputs)
    out = torch._scaled_grouped_mm(a, weights["gate_up"], s_a, weights["s_gate_up"], offs=offs, out_dtype=hidden.dtype)
    out = experts._apply_gate(out)
    a, s_a = quantize_rows(out)
    out = torch._scaled_grouped_mm(a, weights["down"], s_a, weights["s_down"], offs=offs, out_dtype=hidden.dtype)
    weighted = out.index_select(0, rows) * sample_weights[perm].unsqueeze(-1)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.size(0), device=perm.device)
    return weighted[inverse].view(tokens, k, size).sum(dim=1).to(hidden.dtype)
