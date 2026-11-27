"""Recover the vision tower's exact BF16 patch tensor from resized RGB bytes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _expand(pixels, lookup, output, total, TILE: tl.constexpr):
    column = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = column < total
    # Each 16 by 16 channel patch supplies the two consecutive temporal slots.
    source = (column // 512) * 256 + column % 256
    color = tl.load(pixels + source, valid, other=0).to(tl.int32)
    bits = tl.load(lookup + color)
    tl.store(output + column, bits, valid)


def expand_rgb8(pixels, table):
    output = torch.empty((pixels.shape[0], 1536), device=pixels.device, dtype=torch.int16)
    _expand[(triton.cdiv(output.numel(), 512),)](pixels, table, output, output.numel(), 512)
    return output.view(torch.bfloat16)
