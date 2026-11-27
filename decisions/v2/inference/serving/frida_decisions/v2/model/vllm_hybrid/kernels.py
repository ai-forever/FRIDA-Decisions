"""Gated DeltaNet's convolution in a vLLM environment.

A C++ `causal_conv1d` built against a different PyTorch ABI cannot load here.
Without it `transformers` falls back on `F.conv1d`, in the model's type.
FLA's Triton kernel computes in float32 and rounds to the input's type at the end, as the
C++ one does, so it takes the place of `causal_conv1d_fn` in the module the flat executor
reads it from.

It is launched directly: `fla.modules.causal_conv1d.causal_conv1d` goes through an autograd
function, an autotuner and a heuristics layer around the one launch, about 100 us of Python
per mixer. `Heuristics(Autotuner(jit))` is how FLA stacks them, so the jitted kernel
is two `.fn` down, and what the two layers would add is passed here.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from importlib.util import find_spec

import torch
import triton

# Tokens x channels per program, and warps. Tuned for BF16 on an H100.
BT = 64
BD = 64
NUM_WARPS = 8


@lru_cache(maxsize=None)
def _jitted():
    from fla.modules.causal_conv1d import ops

    kernel = ops.causal_conv1d_fwd_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    return kernel


def conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None,
           bd: int = BD, bt: int = BT, num_warps: int = NUM_WARPS) -> torch.Tensor:
    """`(B, T, D)` contiguous in, the same shape out. `flat` has one row; per_option, on the same
    module, has many."""
    rows, tokens, channels = x.shape
    width = weight.shape[1]
    y = torch.empty_like(x)
    _jitted()[(triton.cdiv(channels, bd), triton.cdiv(tokens, bt), rows)](
        x=x, y=y, weight=weight, bias=bias, residual=None, cu_seqlens=None, initial_state=None,
        chunk_indices=None, B=rows, T=tokens, stride_x_n=tokens * channels, stride_x_t=channels, stride_x_d=1,
        D=channels, W=width, BT=bt, BW=triton.next_power_of_2(width), BD=bd, ACTIVATION=activation,
        HAS_WEIGHT=True, HAS_BIAS=bias is not None, HAS_RESIDUAL=False, USE_INITIAL_STATE=False,
        IS_VARLEN=False, num_warps=num_warps)
    return y


def fla_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                  activation: str | None = None, **kwargs) -> torch.Tensor:
    """`causal_conv1d_fn`'s shape, `(B, D, T)` in and out. Off a GPU, FLA's own way in."""
    if not x.is_cuda:
        from fla.modules.causal_conv1d import causal_conv1d

        return causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias, activation=activation)[0].transpose(1, 2)
    return conv1d(x.transpose(1, 2).contiguous(), weight, bias, activation).transpose(1, 2)


def install_conv(cell: torch.nn.Module, use_fla: bool = True) -> str:
    """Returns which convolution the flat executor will call for this mixer's module."""
    kit = sys.modules[type(cell).__module__]
    original = kit.__dict__.setdefault("_transformers_conv1d_fn", kit.causal_conv1d_fn)
    if find_spec("causal_conv1d") is not None:
        return "causal_conv1d"
    if not use_fla or find_spec("fla") is None:
        kit.causal_conv1d_fn = original
        return "F.conv1d"
    kit.causal_conv1d_fn = fla_conv1d_fn
    return "fla"
