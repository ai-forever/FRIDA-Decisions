"""Trainable rows of a frozen embedding table (separators, readout tokens). Any backbone."""
from __future__ import annotations

import torch
from torch import nn


class SpecialRows(nn.Module):
    """A frozen embedding table with a few of its rows trainable.

    The output is the table's row plus a learned delta for the listed ids and
    the table's row alone for every other id. PEFT's trainable tokens do the same
    by rebuilding the whole 93,651 x 1,536 table every forward and taking a dense
    gradient through it, ~0.6 GB that an 8 GB card does not have to spare; this
    is one (tokens x ids) one-hot product. Merging for serving is
    `weight[ids] += offset`, with `offset` what `save_format` writes.

    The trained tensor is in units of the table's own spread: FRIDA's entries have
    a standard deviation of 3.5 (rows of norm 130-200), and Adam's step is absolute,
    so a raw delta at the LoRA learning rate would move these rows hundreds of
    times more slowly, relative to their size, than anything else that trains.
    """

    def __init__(self, base: nn.Embedding, ids: list[int], scale: float | None = None):
        """`scale`: the table's spread as measured when the rows were trained, for a table
        that has no values yet (a serving folder's model is built on `meta`)."""
        super().__init__()
        self.base = base
        self.register_buffer("ids", torch.tensor(list(ids), dtype=torch.long), persistent=False)
        self.scale = float(base.weight.detach().float().std()) if scale is None else float(scale)
        self.delta = nn.Parameter(torch.zeros(len(ids), base.embedding_dim, dtype=torch.float32))

    @property
    def offset(self) -> torch.Tensor:
        """What is added to each listed row, in the table's units."""
        return self.delta * self.scale

    @property
    def weight(self):                    # anything that inspects the table sees the base one
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        hit = (input_ids[..., None] == self.ids).to(self.delta.dtype)
        return (out.float() + hit @ self.offset).to(out.dtype)
