"""Triton kernels that take a hybrid's norms in one launch each, for `CausalFlatExecutor`.

A norm in PyTorch takes about 10 launches, and they were 46% of a flat forward's
(2B at 16 layers, one request). The arithmetic and its order are those of
`Qwen3_5RMSNorm` and `Qwen3_5RMSNormGated`, casts included; only the sum inside the
mean and `rsqrt` may round differently, so a bf16 output can move by one step.
"""
from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources


@triton.jit
def _rms_norm_kernel(x_ptr, scale_ptr, y_ptr, eps, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, BD)
    mask = at < D
    x = tl.load(x_ptr + row * D + at, mask=mask, other=0.0).to(tl.float32)
    y = x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)) * tl.load(scale_ptr + at, mask=mask, other=0.0)
    tl.store(y_ptr + row * D + at, y, mask=mask)


@triton.jit
def _rms_norm_gated_kernel(x_ptr, weight_ptr, gate_ptr, y_ptr, eps, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, BD)
    mask = at < D
    kind = x_ptr.dtype.element_ty
    x = tl.load(x_ptr + row * D + at, mask=mask, other=0.0).to(tl.float32)
    y = (x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps))).to(kind).to(tl.float32)
    y = (tl.load(weight_ptr + at, mask=mask, other=0.0).to(tl.float32) * y).to(kind).to(tl.float32)
    gate = tl.load(gate_ptr + row * D + at, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * D + at, y * (gate * tl.sigmoid(gate)), mask=mask)


# Tuned in bf16 on H100. An autotuner benchmarks every config per new key, tens of
# seconds on the first batches, and its Python takes most of a chunked kernel's launch.
PINNED = {
    "chunk_fwd_kernel_o": dict(kwargs={"BK": 32, "BV": 32}, num_warps=2),
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": dict(kwargs={"BV": 32}, num_warps=4),
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": dict(kwargs={"BK": 64}, num_warps=2),
    "chunk_local_cumsum_scalar_kernel": dict(kwargs={}, num_warps=1),
    "l2norm_fwd_kernel": dict(kwargs={"BT": 32}, num_warps=16),
    "recompute_w_u_fwd_kernel": dict(kwargs={}, num_warps=4),
}


class _Pinned(triton.KernelInterface):
    """An autotuned kernel at one config. A config too large for this dtype's shared
    memory (fp32) goes back to the autotuner."""

    def __init__(self, tuner: Autotuner, config: dict):
        self.tuner, self.fn = tuner, tuner.fn
        self.config = {**config["kwargs"], "num_warps": config["num_warps"], "num_stages": 3, "num_ctas": 1}
        self.falls_back = False

    def run(self, *args, **kwargs):
        if self.falls_back:
            return self.tuner.run(*args, **kwargs)
        try:
            return self.fn.run(*args, **kwargs, **self.config)
        except OutOfResources:
            self.falls_back = True
            return self.tuner.run(*args, **kwargs)


def _kernel_name(tuner: Autotuner) -> str:
    fn = tuner.fn
    while not hasattr(fn, "__name__"):             # `triton.heuristics` around the kernel
        fn = fn.fn
    return fn.__name__


def _tuners():
    for name, module in list(sys.modules.items()):
        if module is not None and name.split(".")[0] == "fla":
            for attribute, tuner in list(vars(module).items()):
                if isinstance(tuner, Autotuner):
                    yield module, attribute, tuner


def pin_fla_kernels() -> None:
    """Swap FLA's autotuned kernels in `PINNED` where its code looks them up."""
    for module, attribute, tuner in _tuners():
        config = PINNED.get(_kernel_name(tuner))
        if config is not None:
            setattr(module, attribute, _Pinned(tuner, config))


def tuned_kernels() -> dict[str, int]:
    """Kernels left to the autotuner, with the keys they have tuned so far."""
    return {_kernel_name(tuner): len(getattr(tuner, "_cache", ())) for _, _, tuner in _tuners()}


@triton.jit(do_not_specialize=["channel_step", "token_step"])
def _wave_kernel(mixed, projected, log_rate, shift, tokens, reads, writes, core, h0, ht, starts, scale,
                 channel_step, token_step,
                 H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                 WIDTH: tl.constexpr, AT_DECAY: tl.constexpr, AT_BETA: tl.constexpr,
                 USE_H0: tl.constexpr, STORE_HT: tl.constexpr):
    """`fused_recurrent_gated_delta_rule_fwd_kernel` of FLA with its gate, sigmoid and L2 norm in
    the kernel, for a wave read in place. A row takes its tokens through `reads` from the
    convolution's output `mixed` `(channels, tokens)`, whatever its strides, and through `tokens` from `projected`;
    `writes` says where a token's output goes in `core`, or -1 for a token read again."""
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(starts + i_n).to(tl.int64), tl.load(starts + i_n + 1).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H0:
        b_h += tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=mask_h, other=0).to(tl.float32)
    b_rate = exp(tl.load(log_rate + i_hv).to(tl.float32))
    b_shift = tl.load(shift + i_hv).to(tl.float32)
    for t in tl.range(bos, eos):
        at = tl.load(reads + t).to(tl.int64)
        token = tl.load(tokens + t).to(tl.int64)
        to = tl.load(writes + t).to(tl.int64)
        at = at * token_step
        b_q = tl.load(mixed + (i_h * K + o_k) * channel_step + at, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(mixed + ((H + i_h) * K + o_k) * channel_step + at, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(mixed + (2 * H * K + i_hv * V + o_v) * channel_step + at, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.sigmoid(tl.load(projected + token * WIDTH + AT_BETA + i_hv).to(tl.float32))
        b_g = tl.load(projected + token * WIDTH + AT_DECAY + i_hv).to(tl.float32)
        b_h *= exp(-b_rate * softplus(b_g + b_shift))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(core + (to * HV + i_hv) * V + o_v, b_o, mask=mask_v & (to >= 0))
    if STORE_HT:
        tl.store(ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :], b_h, mask=mask_h)


def recurrent_wave(mixed, projected, log_rate, shift, tokens, reads, writes, core, start, keep, starts,
                   heads_k, heads_v, k, v, width, at_decay, at_beta):
    """One wave in one launch, its output written into `core` `(N, heads_v, v)`. Returns the rows'
    final states if `keep`."""
    rows = starts.numel() - 1
    bv = min(8, triton.next_power_of_2(v))
    last = torch.empty(rows, heads_v, k, v, dtype=torch.float32, device=core.device) if keep else None
    _wave_kernel[(triton.cdiv(v, bv) * rows * heads_v,)](
        mixed, projected, log_rate, shift, tokens, reads, writes, core, start, last, starts, k ** -0.5,
        mixed.stride(0), mixed.stride(1),
        H=heads_k, HV=heads_v, K=k, V=v, BK=triton.next_power_of_2(k), BV=bv, WIDTH=width, AT_DECAY=at_decay,
        AT_BETA=at_beta, USE_H0=start is not None, STORE_HT=keep, num_warps=1, num_stages=3)
    return last


def _block(size: int) -> int:
    """The next power of two: `tl.arange` takes nothing else."""
    return 1 << (size - 1).bit_length()


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """`Qwen3_5RMSNorm` over the last dimension of a contiguous `x`. `scale` is `1 + weight` in float32."""
    size = x.shape[-1]
    if not x.is_contiguous():
        raise ValueError("a norm's kernel reads packed memory")
    out = torch.empty_like(x)
    _rms_norm_kernel[(x.numel() // size,)](x, scale, out, eps, size, _block(size))
    return out


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`Qwen3_5RMSNormGated` over the last dimension, its activation being SiLU."""
    size = x.shape[-1]
    if not (x.is_contiguous() and gate.is_contiguous()):
        raise ValueError("a norm's kernel reads packed memory")
    out = torch.empty_like(x)
    _rms_norm_gated_kernel[(x.numel() // size,)](x, weight, gate, out, eps, size, _block(size))
    return out
