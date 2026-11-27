"""The scalar head, shared by every backbone: pool the read tokens per slot, then `w.h + b`.

A slot is one option. Its read tokens are either the option's whole span (mean
readout, layout 1) or its last token (layout 2, and every causal backbone); a
mean over one token is that token, exactly, so one head serves both.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Readout:
    """Where the scalars are read: token `(row[t], col[t])` contributes to `slot[t]`."""
    row: torch.Tensor      # (T,)
    col: torch.Tensor      # (T,)
    slot: torch.Tensor     # (T,)
    n_slots: int | None = None

    @classmethod
    def from_spans(cls, spans: list[tuple[int, int, int]], n_slots: int | None = None) -> "Readout":
        """`spans[k] = (row, start, end)` of slot k, in slot order."""
        rows, cols, slots = [], [], []
        for slot, (r, start, end) in enumerate(spans):
            rows += [r] * (end - start)
            cols += list(range(start, end))
            slots += [slot] * (end - start)
        return cls(torch.tensor(rows), torch.tensor(cols), torch.tensor(slots),
                   len(spans) if n_slots is None else n_slots)

    def to(self, device) -> "Readout":
        return Readout(self.row.to(device), self.col.to(device), self.slot.to(device), self.n_slots)


def scatter_mean(hidden: torch.Tensor, readout: Readout) -> torch.Tensor:
    """(n_slots, D) float32: the mean of each slot's read tokens."""
    picked = hidden[readout.row, readout.col].float()
    slots = readout.n_slots if readout.n_slots is not None else int(readout.slot.max()) + 1
    pooled = torch.zeros(slots, picked.shape[-1], device=picked.device)
    counts = torch.zeros(slots, 1, device=picked.device)
    pooled.index_add_(0, readout.slot, picked)
    counts.index_add_(0, readout.slot, torch.ones_like(picked[:, :1]))
    return pooled / counts


def scalar_head(d_model: int) -> nn.Linear:
    """The trained part: fp32, zero bias. Saved as `head.pt` (its state dict)."""
    head = nn.Linear(d_model, 1, dtype=torch.float32)
    nn.init.zeros_(head.bias)
    return head


class LinearReadoutHead:
    """`Head` over a `nn.Linear` owned by the model (so `head.pt` keeps its keys)."""

    def __init__(self, linear: nn.Linear):
        self.linear = linear

    def __call__(self, hidden: torch.Tensor, readout: Readout) -> torch.Tensor:
        return self.linear(scatter_mean(hidden, readout)).squeeze(-1)
