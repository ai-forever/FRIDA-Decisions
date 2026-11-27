"""Fuse SiLU and multiplication while preserving the BF16 activation rounding point."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(src, lut, out, count, I: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    src_off = (off // I) * (2 * I) + off % I
    raw = tl.load(src + src_off, off < count, 0.).to(tl.uint16, bitcast=True)
    activated = tl.load(lut + raw.to(tl.int32)).to(tl.float32)
    up = tl.load(src + src_off + I, off < count, 0.).to(tl.float32)
    tl.store(out + off, activated * up, off < count)


@lru_cache(maxsize=8)
def _silu_table(device):
    bits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
    return torch.nn.functional.silu(bits.view(torch.bfloat16))


def gated(projected):
    doubled = projected.shape[-1]
    inner = doubled // 2
    out = projected.new_empty((*projected.shape[:-1], inner))
    tile = 512
    _gate_up[(triton.cdiv(out.numel(), tile),)](projected, _silu_table(projected.device), out, out.numel(), inner, tile)
    return out
