"""The segment tree a renderer produces, and the one row layout every executor uses.

A request is a tree `state -> question -> option`. Each segment sees itself and
its ancestors and nothing else; options never see each other. A renderer turns
a compiled request into a `ScoringPlan` (segments with token ids, positions and,
on the leaves, where the head reads); an executor decides how the leaves travel
through the backbone. This module is backbone-agnostic and has no torch.

`layout_rows` is the single implementation of the row layout: which leaves share
a row, where a row is split (`max_options` leaves, or the next leaf would take it
past `max_tokens`), and where every block lands inside the row. Rows hold every
segment except the root; the root (the state) is either repeated at the head of
each row (dense packing: `budget_offset = len(root)`) or computed once and
attended to from outside (state cache: `budget_offset = 0`). Splitting is exact:
which row a leaf travels in cannot change its score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

READOUTS = ("mean", "last")


@dataclass
class Segment:
    """One node of the tree.

    `pos` are the backbone positions of the segment's tokens; position restarts
    (every option of a question at the same positions, right after the question)
    are encoded here, not by the executor. `readout` is `(slot, "mean"|"last")`
    on leaves the head reads from: the mean over the segment's tokens, or its
    last token.
    """
    parent: int | None
    ids: list[int]
    pos: Sequence[int]
    kind: Literal["state", "question", "option"]
    readout: tuple[int, str] | None = None
    cache_key: bytes | None = None

    def read_span(self) -> tuple[int, int]:
        """Read tokens as a `[start, end)` range of this segment's own tokens."""
        mode = self.readout[1]
        if mode == "mean":
            return 0, len(self.ids)
        if mode == "last":
            return len(self.ids) - 1, len(self.ids)
        raise ValueError(f"readout {mode!r} is not one of {READOUTS}")


@dataclass
class ScoringPlan:
    """What a renderer hands an executor: the tree of one request.

    Slots are numbered in option order (the `Compiled.flat()` order, i.e. the
    margin order); `group[slot]` is the index of the slot's question within the
    plan and `keys[slot]` its `(question_id, option_id)`.
    """
    segments: list[Segment]
    n_slots: int
    group: list[int]
    n_groups: int
    keys: list[tuple[str, str]] = field(default_factory=list)
    mm: dict[str, Any] = field(default_factory=dict)   # media tensors, later

    @property
    def root(self) -> Segment:
        return self.segments[0]

    def leaves(self) -> list[int]:
        """Indices of the read segments, in slot order."""
        out = [i for i, s in enumerate(self.segments) if s.readout is not None]
        out.sort(key=lambda i: self.segments[i].readout[0])
        return out

    def chain(self, index: int) -> list[int]:
        """Ancestors of a segment below the root, root-side first, itself last."""
        out = []
        while index is not None and self.segments[index].parent is not None:
            out.append(index)
            index = self.segments[index].parent
        return out[::-1]


@dataclass
class Row:
    """Every segment of a row but the root, in row-local columns.

    `blocks[k] = (start, end, segment)`; `reads[k] = (start, end, slot)` with the
    plan's slot. Positions are absolute (they already count the root).
    """
    ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    blocks: list[tuple[int, int, int]] = field(default_factory=list)
    reads: list[tuple[int, int, int]] = field(default_factory=list)
    spans: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def n_reads(self) -> int:
        return len(self.reads)


def validate(plan: ScoringPlan) -> None:
    """A plan the row layout can lay out: one root first, parents before children."""
    if not plan.segments or plan.segments[0].parent is not None:
        raise ValueError("segment 0 must be the root")
    for i, s in enumerate(plan.segments[1:], 1):
        if s.parent is None or not 0 <= s.parent < i:
            raise ValueError(f"segment {i}: parent {s.parent} must precede it")
        if len(s.pos) != len(s.ids):
            raise ValueError(f"segment {i}: {len(s.ids)} ids, {len(s.pos)} positions")
    slots = sorted(s.readout[0] for s in plan.segments if s.readout is not None)
    if slots != list(range(plan.n_slots)) or len(plan.group) != plan.n_slots:
        raise ValueError("readout slots must be 0..n_slots-1, one group per slot")


def layout_rows(plan: ScoringPlan, budget_offset: int, max_options: int,
                max_tokens: int) -> list[Row]:
    """Split a plan's leaves into rows, repeating their ancestors in each row.

    A row is closed when it holds `max_options` leaves, or when the next leaf
    plus whichever of its ancestors the row does not hold yet would take
    `budget_offset + len(row)` past `max_tokens` -- unless the row has no leaf
    yet: one leaf always fits, however long.
    """
    segments = plan.segments
    rows, row = [], Row()
    for leaf in plan.leaves():
        chain = plan.chain(leaf)
        pending = sum(len(segments[a].ids) for a in chain[:-1] if a not in row.spans)
        body = len(segments[leaf].ids)
        if row.n_reads >= max_options or (
                row.n_reads and budget_offset + len(row.ids) + pending + body > max_tokens):
            rows.append(row)                        # full; continue in a new one
            row = Row()
        for a in chain:                             # (re-)emit what the row lacks
            if a in row.spans:
                continue
            start = len(row.ids)
            row.ids += segments[a].ids
            row.positions += list(segments[a].pos)
            row.spans[a] = (start, len(row.ids))
            row.blocks.append((start, len(row.ids), a))
        start = row.spans[leaf][0]
        lo, hi = segments[leaf].read_span()
        row.reads.append((start + lo, start + hi, segments[leaf].readout[0]))
    if row.reads:
        rows.append(row)
    return rows


def visible(plan: ScoringPlan, row: Row, segment: int) -> list[tuple[int, int]]:
    """Row-local column ranges a block may attend to, root excluded: its
    ancestors present in the row, and itself."""
    return [row.spans[a] for a in plan.chain(segment)]


def aligned(width: int, align: int) -> int:
    return -(-width // align) * align
