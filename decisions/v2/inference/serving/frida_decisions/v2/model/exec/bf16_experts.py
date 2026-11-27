"""BF16 grouped experts with a fused output reorder, routing weight and sum."""
import torch
import triton
import triton.language as tl


@triton.jit
def _combine(projections, inverse, weights, output, D: tl.constexpr, K: tl.constexpr,
             TILE: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    # ATen's BF16 sum across eight strided rows uses four FP32 accumulators.
    total = tl.full((TILE,), 0, tl.float32)
    for lane in tl.static_range(4):
        partial = tl.full((TILE,), 0, tl.float32)
        for offset in tl.static_range(2):
            index = token * K + lane + 4 * offset
            row = tl.load(inverse + index).to(tl.int64)
            weight = tl.load(weights + index).to(tl.float32)
            value = tl.load(projections + row * D + col, col < D, 0.).to(tl.float32)
            partial += (value * weight).to(tl.bfloat16).to(tl.float32)
        total += partial
    tl.store(output + token * D + col, total, col < D)


def grouped_experts(experts, hidden, chosen, weights):
    from transformers.integrations.moe import _grouped_linear
    if hidden.dtype != torch.bfloat16 or weights.dtype != hidden.dtype or not hidden.is_contiguous():
        raise ValueError("Expected contiguous BF16 activations and BF16 routing weights")
    if experts.has_bias or experts.is_transposed or experts._is_expert_parallel or not experts.has_gate:
        raise ValueError("Expected gated unsharded Qwen experts")
    tokens, dim = hidden.shape
    topk = chosen.shape[1]
    assert topk == 8 and dim == 2048
    ids, order = torch.sort(chosen.reshape(-1))
    counts = torch.histc(ids.int(), bins=experts.num_experts, min=0, max=experts.num_experts - 1)
    offsets = counts.cumsum(0, dtype=torch.int32)
    projected = _grouped_linear(hidden[order // topk], experts.gate_up_proj, offsets)
    if experts.config.hidden_act == "silu" and experts.is_concatenated:
        from frida_decisions.v2.model.exec.bf16_gate import gated
        activated = gated(projected)
    else:
        activated = experts._apply_gate(projected)
    down = _grouped_linear(activated, experts.down_proj, offsets)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    result = torch.empty_like(hidden)
    tile = 256
    _combine[(tokens, triton.cdiv(dim, tile))](down, inverse, weights, result, dim, topk, tile)
    return result
