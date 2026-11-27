"""Causal-LM executors for a `ScoringPlan`: per-option sequences, and the tree mask.

All are exact implementations of the same visibility (an option sees the
state, its question and itself, causally), so they give the same margins up to
floating point (`tests/test_causal_scalar.py`):

* `CausalPerOptionExecutor` -- one ordinary causal sequence per option,
  `state + question + option`, batched with right padding. The simplest exact
  version: any HF causal LM, any attention kernel (sdpa / flash), no custom
  mask. The state is recomputed per option.
* `CausalTreeExecutor` -- several options per row on the shared prefix, with a
  4D additive mask (`allowed(i,j) = seg(j) in anc*(seg(i)) and (seg(j) != seg(i)
  or j <= i)`) and positions restarting per option. Rows come from the same
  `layout.layout_rows` as the T5 path. This is the reference for the
  prefix-sharing executors to come (FlexAttention `BlockMask` from the same
  segment ids, or prefix KV + batched suffixes): they must reproduce its numbers.
  Not for FlashAttention-varlen: it reads a reset of `position_ids` as a new
  document.
* `CausalPrefixExecutor` -- the tree level by level over a KV cache: the state
  once per request, each question once, the options batched on top. The same
  visibility as the tree mask without an `(L, L)` mask, and nothing computed
  twice; any attention kernel that takes a 2D padding mask with a cache.

Token-budget micro-batching (`max_batch_tokens`): sequences are sorted by
length and grouped so that rows x padded length stays under the budget; the
margins come back in slot order either way. `None` runs one forward.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import torch

from frida_decisions.v2.model.head import Readout
from frida_decisions.v2.model.layout import ScoringPlan, layout_rows, visible


@dataclass
class _Sequence:
    ids: list[int]
    pos: list[int]
    reads: list[tuple[int, int, int]]          # (start, end, global slot)
    mask: torch.Tensor | None = None           # (n, n) bool, None = plain causal
    pos3: torch.Tensor | None = None           # (3, n) M-RoPE positions, a state with images
    media: dict | None = None                  # its plan's `mm` (pixel_values, image_grid_thw)


@dataclass
class CausalMicroBatch:
    input_ids: torch.Tensor                    # (b, L)
    position_ids: torch.Tensor                 # (b, L)
    attention_mask: torch.Tensor               # (b, L) long, or (b, 1, L, L) bool (tree)
    readout: Readout                           # rows / cols local, slots 0..m-1
    slots: torch.Tensor                        # (m,) global slot of each local slot
    # Rows with images, in row order: (row, its plan's `mm`); position_ids is
    # then (3, b, L). Rows of one plan share the dict (its images are encoded once).
    media: list[tuple[int, dict]] = field(default_factory=list)

    def to(self, device):
        moved = {}
        for _, mm in self.media:
            if id(mm) not in moved:
                moved[id(mm)] = {k: v.to(device) for k, v in mm.items()}
        return CausalMicroBatch(self.input_ids.to(device), self.position_ids.to(device),
                                self.attention_mask.to(device), self.readout.to(device),
                                self.slots.to(device),
                                [(row, moved[id(mm)]) for row, mm in self.media])


@dataclass
class CausalBatch:
    """What a causal executor packs: micro-batches, and `group` like `Packed.group`."""
    micro: list[CausalMicroBatch]
    group: torch.Tensor                        # (N,) question of each slot, across the batch
    lengths: list[int]                         # unpadded length of every sequence / row
    n_slots: int

    def to(self, device):
        return CausalBatch([m.to(device) for m in self.micro], self.group.to(device),
                           self.lengths, self.n_slots)

    @property
    def padded_tokens(self) -> int:
        return sum(int(m.input_ids.numel()) for m in self.micro)


def _bases(plans: Sequence[ScoringPlan]):
    slot_base, group, s_acc, g_acc = [], [], 0, 0
    for plan in plans:
        slot_base.append(s_acc)
        group += [g_acc + g for g in plan.group]
        s_acc += plan.n_slots
        g_acc += plan.n_groups
    return slot_base, torch.tensor(group, dtype=torch.long)


# One pass through the backbone costs about as much as this many padded tokens: 23 ms of
# launches against 3 us a token on an H100 (16-layer 2B).
PASS_TOKENS = 8192


def _micro_groups(lengths: list[int], max_batch_tokens: int | None) -> list[list[int]]:
    """Indices per micro-batch. Without a budget: everything, in order."""
    if max_batch_tokens is None:
        return [list(range(len(lengths)))]
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups, current, width = [], [], 0
    for i in order:
        w = max(width, lengths[i])
        if current and (len(current) + 1) * w > max_batch_tokens:
            groups.append(current)
            current, w = [], lengths[i]
        current.append(i)
        width = w
    if current:
        groups.append(current)
    return groups


def _padded_groups(lengths: list[int], max_batch_tokens: int | None) -> list[list[int]]:
    """Indices per micro-batch of rows padded to their longest: a new one when the
    pads would cost more than a pass (`PASS_TOKENS`) or the budget would overflow.
    Without a budget: everything, in order, like `_micro_groups`."""
    if max_batch_tokens is None:
        return [list(range(len(lengths)))]
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups, current, wasted = [], [], 0
    for i in order:
        width = lengths[current[0]] if current else lengths[i]
        over = wasted + width - lengths[i] > PASS_TOKENS
        if current and (over or (len(current) + 1) * width > max_batch_tokens):
            groups.append(current)
            current, wasted, width = [], 0, lengths[i]
        current.append(i)
        wasted += width - lengths[i]
    if current:
        groups.append(current)
    return groups


def _collate(seqs: list[_Sequence], pad_id: int, tree: bool) -> CausalMicroBatch:
    width = max(len(s.ids) for s in seqs)
    b = len(seqs)
    input_ids = torch.full((b, width), pad_id, dtype=torch.long)
    m_rope = any(s.pos3 is not None for s in seqs)
    position_ids = torch.zeros((3, b, width) if m_rope else (b, width), dtype=torch.long)
    media = []
    if tree:
        mask = torch.zeros((b, 1, width, width), dtype=torch.bool)
    else:
        mask = torch.zeros((b, width), dtype=torch.long)
    spans, slots = [], []
    for i, s in enumerate(seqs):
        n = len(s.ids)
        input_ids[i, :n] = torch.tensor(s.ids)
        if m_rope:
            position_ids[:, i, :n] = (s.pos3 if s.pos3 is not None
                                      else torch.tensor(s.pos).expand(3, -1))
            if s.media:
                media.append((i, s.media))
        else:
            position_ids[i, :n] = torch.tensor(s.pos)
        if tree:
            mask[i, 0, :n, :n] = s.mask
            pads = torch.arange(n, width)
            mask[i, 0, pads, pads] = True          # a pad must attend to something; nothing reads it
        else:
            mask[i, :n] = 1
        for start, end, slot in s.reads:
            spans.append((i, start, end))
            slots.append(slot)
    return CausalMicroBatch(input_ids, position_ids, mask, Readout.from_spans(spans),
                            torch.tensor(slots, dtype=torch.long), media)


CONFIG = object()        # "take it from the model's config"


def text_only(plans, executor, allowed: bool = False):
    """Plans with images (`plan.mm`) are read by per_option (one plain causal
    sequence per option, positions the model's own M-RoPE) and flat, not by these."""
    if not allowed and any(getattr(p, "mm", None) for p in plans):
        raise ValueError(f"{type(executor).__name__} does not read images; use per_option or flat")


class _CausalExecutor:
    """Shared by both: micro-batching, the forward and the head.

    `model` provides `backbone` (an HF base model returning `last_hidden_state`),
    `readout_head`, `pad_token_id` and `config` (a `CausalScalarConfig`); read at
    call time, so a PEFT wrapper can be swapped in."""
    tree = False
    # No KV cache: transformers' per-layer checkpointing applies as is.
    trains_with_checkpointing = True

    def __init__(self, model, max_batch_tokens=CONFIG):
        self.model = model
        self._budget = max_batch_tokens

    @property
    def max_batch_tokens(self) -> int | None:
        """The micro-batch budget: the model config's unless given; None = off."""
        if self._budget is CONFIG:
            return self.model.config.max_batch_tokens
        return self._budget

    def sequences(self, plans: Sequence[ScoringPlan]) -> list[_Sequence]:
        raise NotImplementedError

    def pack(self, plans: Sequence[ScoringPlan]) -> CausalBatch:
        text_only(plans, self, allowed=not self.tree)
        seqs = self.sequences(plans)
        lengths = [len(s.ids) for s in seqs]
        micro = [_collate([seqs[i] for i in idx], self.model.pad_token_id, self.tree)
                 for idx in _micro_groups(lengths, self.max_batch_tokens)]
        _, group = _bases(plans)
        return CausalBatch(micro, group, lengths, sum(p.n_slots for p in plans))

    def cost(self, plans: Sequence[ScoringPlan]) -> int:
        """Padded tokens of the forward (attention is linear-memory with sdpa)."""
        lengths = [len(s.ids) for s in self.sequences(plans)]
        return sum(len(idx) * max(lengths[i] for i in idx)
                   for idx in _micro_groups(lengths, self.max_batch_tokens))

    def hidden(self, mb: CausalMicroBatch) -> torch.Tensor:
        backbone = self.model.backbone
        mask = mb.attention_mask
        if mb.media:
            return self._hidden_with_images(mb)
        if mask.dim() == 4:                        # additive, in the activations' dtype
            # The embedding table's, not the first parameter's: trainable LoRA
            # and separator rows are fp32 on a bf16 backbone, and CUDA SDPA
            # refuses a mask whose dtype differs from the query's.
            dtype = backbone.get_input_embeddings().weight.dtype
            mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(
                ~mask, torch.finfo(dtype).min)
        return backbone(input_ids=mb.input_ids, attention_mask=mask,
                        position_ids=mb.position_ids, use_cache=False).last_hidden_state

    def _hidden_with_images(self, mb: CausalMicroBatch) -> torch.Tensor:
        """As `Qwen3_5Model.forward`: the token embeddings with the image
        features in place of the image-pad tokens (row-major order: rows in
        order, a row's images in order), then the text model on embeddings and
        (3, b, L) M-RoPE positions. A plan's images are encoded once."""
        vision = self.model.vision
        backbone = self.model.backbone
        embeds = backbone.get_input_embeddings()(mb.input_ids)
        done, parts = {}, []
        for _, mm in mb.media:
            if id(mm) not in done:
                done[id(mm)] = vision.features(mm)
            parts.append(done[id(mm)])
        image = torch.cat(parts).to(embeds.device, embeds.dtype)
        where = (mb.input_ids == vision.image_token_id).unsqueeze(-1)
        if int(where.sum()) * embeds.shape[-1] != image.numel():
            raise RuntimeError(f"{int(where.sum())} image tokens for {image.shape[0]} image features")
        embeds = embeds.masked_scatter(where, image)
        return backbone(inputs_embeds=embeds, attention_mask=mb.attention_mask,
                        position_ids=mb.position_ids, use_cache=False).last_hidden_state

    def forward(self, batch: CausalBatch) -> torch.Tensor:
        """One margin per slot, in slot order (the `Compiled.flat()` order)."""
        head = self.model.readout_head
        scores, slots = [], []
        for mb in batch.micro:
            scores.append(head(self.hidden(mb), mb.readout))
            slots.append(mb.slots)
        scores, slots = torch.cat(scores), torch.cat(slots)
        return scores[torch.argsort(slots)]

    def score(self, plans: Sequence[ScoringPlan]) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        margins = self.forward(self.pack(plans).to(device))
        return list(margins.split([p.n_slots for p in plans]))


class CausalPerOptionExecutor(_CausalExecutor):
    """One `state + question + option` sequence per option."""
    tree = False

    def sequences(self, plans):
        slot_base, _ = _bases(plans)
        out = []
        for p, plan in enumerate(plans):
            for leaf in plan.leaves():
                images = bool(plan.mm)
                ids = list(plan.root.ids)
                pos = list(range(len(ids))) if images else list(plan.root.pos)
                tail = []
                for a in plan.chain(leaf):
                    ids += plan.segments[a].ids
                    pos += list(plan.segments[a].pos)
                    tail += list(plan.segments[a].pos)
                lo, hi = plan.segments[leaf].read_span()
                start = len(ids) - len(plan.segments[leaf].ids)
                pos3 = (torch.cat([plan.root.pos, torch.tensor(tail, dtype=torch.long).expand(3, -1)],
                                  dim=1) if images else None)
                out.append(_Sequence(ids, pos, [(start + lo, start + hi,
                                                 slot_base[p] + plan.segments[leaf].readout[0])],
                                     pos3=pos3, media=plan.mm if images else None))
        return out


class CausalTreeExecutor(_CausalExecutor):
    """Rows of `state | question | options...` with the tree mask (see the module doc)."""
    tree = True

    def sequences(self, plans):
        c = self.model.config
        max_tokens = c.max_row_tokens if c.max_row_tokens is not None else 10**12
        slot_base, _ = _bases(plans)
        out = []
        for p, plan in enumerate(plans):
            s = len(plan.root.ids)
            for row in layout_rows(plan, s, c.max_options_per_row, max_tokens):
                n = s + len(row.ids)
                mask = torch.zeros((n, n), dtype=torch.bool)
                causal = torch.ones((n, n), dtype=torch.bool).tril()
                mask[:s, :s] = causal[:s, :s]
                for start, end, segment in row.blocks:
                    a0, a1 = s + start, s + end
                    mask[a0:a1, :s] = True
                    for v0, v1 in visible(plan, row, segment)[:-1]:   # ancestors, whole
                        mask[a0:a1, s + v0:s + v1] = True
                    mask[a0:a1, a0:a1] = causal[:a1 - a0, :a1 - a0]  # itself, causally
                out.append(_Sequence(plan.root.ids + row.ids, list(plan.root.pos) + row.positions,
                                     [(s + a, s + b, slot_base[p] + slot)
                                      for a, b, slot in row.reads], mask))
        return out


@dataclass
class PrefixMicroBatch:
    """One forward of a level: its rows' own tokens, where their ancestors sit
    in the pools above, and which of their tokens the level's pool keeps."""
    input_ids: torch.Tensor                    # (b, w)
    position_ids: torch.Tensor                 # (b, w)
    attention_mask: torch.Tensor               # (b, ancestors' widths + w) long
    index: list[torch.Tensor]                  # per depth above: (b, width) positions in its pool
    writes: tuple[torch.Tensor, ...] | None    # (row, column, pool position) per kept token
    readout: Readout | None                    # rows / cols local, slots 0..m-1
    slots: torch.Tensor | None                 # (m,) global slot of each local slot
    parents: torch.Tensor | None = None        # (b,) each row's parent among the level above's nodes
    kept: tuple[torch.Tensor, ...] | None = None  # (row, node) per kept segment

    def to(self, device):
        return PrefixMicroBatch(
            self.input_ids.to(device), self.position_ids.to(device),
            self.attention_mask.to(device), [ix.to(device) for ix in self.index],
            None if self.writes is None else tuple(t.to(device) for t in self.writes),
            None if self.readout is None else self.readout.to(device),
            None if self.slots is None else self.slots.to(device),
            None if self.parents is None else self.parents.to(device),
            None if self.kept is None else tuple(t.to(device) for t in self.kept))


@dataclass
class PrefixBatch:
    """What `CausalPrefixExecutor` packs: per level, its pool's size in tokens
    and in segments, and its micro-batches; `group` like `CausalBatch.group`."""
    levels: list[tuple[int, int, list[PrefixMicroBatch]]]
    group: torch.Tensor
    n_slots: int

    def to(self, device):
        return PrefixBatch([(tokens, nodes, [m.to(device) for m in micro])
                            for tokens, nodes, micro in self.levels],
                           self.group.to(device), self.n_slots)


class CausalPrefixExecutor:
    """The segment tree level by level over a KV cache: every segment computed once.

    Level 0 is the roots (states), level 1 their questions, level 2 the options.
    A level is one forward (micro-batched by `max_batch_tokens`) whose rows read
    their ancestors' keys and values: a state is encoded once per request and a
    question once, however many options hang below it, and no `(L, L)` mask is
    built. The visibility is `CausalTreeExecutor`'s -- a row sees its ancestors
    whole and itself causally, at the renderer's positions -- so the margins
    are the same up to floating point (`tests/test_causal_scalar.py`).

    Memory. A segment with children keeps its own keys and values once, in its
    level's pool, written in place by the level's forward (keys are cached after
    RoPE, so positions travel with them).
    The cache a row attends through (`_GatherLayer`) stores nothing: on every
    call it gathers the row's ancestors from the pools by index, pads them to the
    micro-batch's longest (masked out by the 2D attention mask) and appends the
    row's own keys and values. Per-row copies of the ancestors therefore exist
    only inside one layer's attention, and a recompute repeats the gather
    exactly. That is what lets transformers' per-layer gradient checkpointing
    keep the cache (`_can_checkpoint_with_cache`, set for the call): the trainer
    turns it on, and without the flag it drops `past_key_values` in train mode
    -- options would silently lose their state and question, so every forward
    checks that each layer read its ancestors.

    Launches. On a fast GPU a small model is bound by kernel launches, not
    FLOPs, and each micro-batch is a whole pass through the backbone: three
    levels cost about three per-option passes of launches (measured on an
    H100 in training: 2.4x slower than per_option at 4 requests a forward).
    `pack` therefore lays out every level on the host and `forward` only
    launches, a fixed number of kernels per layer and micro-batch however many
    rows it holds (one gather per depth, one pool write); the saving in FLOPs
    pays off from larger batches up.

    Recurrent layers (a hybrid backbone). A Gated DeltaNet layer reads no keys:
    what a child needs is its parent's final recurrent state and the last tokens
    of its convolution (`_ForkLayer`), kept once per segment and copied per row.
    Pads would decay the recurrent state, so a segment with children and a parent
    runs in a micro-batch of its own length, without pads. A root starts from a
    zero state and a pad's keys are exactly zero, so left pads leave it at zero and
    the roots share a micro-batch (`_padded_groups`); the options keep right
    padding, which nothing reads. Inference only: the states are not checkpoint-safe.
    """
    # The trainer accepts this executor under gradient checkpointing.
    trains_with_checkpointing = True

    def __init__(self, model, max_batch_tokens=CONFIG):
        self.model = model
        self._budget = max_batch_tokens

    @property
    def max_batch_tokens(self) -> int | None:
        if self._budget is CONFIG:
            return self.model.config.max_batch_tokens
        return self._budget

    # ------------------------------------------------------------ layout
    @staticmethod
    def _levels(plans: Sequence[ScoringPlan]) -> list[list[tuple[int, int]]]:
        """(plan, segment) per level, roots first; parents always on the level above."""
        levels, current = [], [(p, 0) for p in range(len(plans))]
        children = [{} for _ in plans]
        for p, plan in enumerate(plans):
            for i, seg in enumerate(plan.segments):
                if seg.parent is not None:
                    children[p].setdefault(seg.parent, []).append(i)
        while current:
            levels.append(current)
            current = [(p, j) for p, i in current for j in children[p].get(i, ())]
        return levels

    @staticmethod
    def _ancestors(plan: ScoringPlan, index: int) -> list[int]:
        """A segment's ancestors, root first."""
        out, parent = [], plan.segments[index].parent
        while parent is not None:
            out.append(parent)
            parent = plan.segments[parent].parent
        return out[::-1]

    def _past(self, plan: ScoringPlan, index: int) -> int:
        return sum(len(plan.segments[a].ids) for a in self._ancestors(plan, index))

    @property
    def _recurrent(self) -> bool:
        return "linear_attention" in (getattr(self.model.backbone.config, "layer_types", None) or ())

    def _micro(self, plans, level) -> list[list[int]]:
        widths = [self._past(plans[p], i) + len(plans[p].segments[i].ids) for p, i in level]
        if not self._recurrent:
            return _micro_groups(widths, self.max_batch_tokens)
        parents = {(p, s.parent) for p, plan in enumerate(plans)
                   for s in plan.segments if s.parent is not None}
        same: dict[int, list[int]] = {}                     # own length if kept, 0 for the leaves
        roots: list[int] = []                               # kept roots, left padded
        for j, (p, i) in enumerate(level):
            if (p, i) not in parents:
                same.setdefault(0, []).append(j)
            elif plans[p].segments[i].parent is None:
                roots.append(j)
            else:
                same.setdefault(len(plans[p].segments[i].ids), []).append(j)
        groups = [[idx[g] for g in group] for idx in same.values()
                  for group in _micro_groups([widths[j] for j in idx], self.max_batch_tokens)]
        if roots:
            groups += [[roots[g] for g in group]
                       for group in _padded_groups([widths[j] for j in roots], self.max_batch_tokens)]
        return groups

    def pack(self, plans: Sequence[ScoringPlan]) -> PrefixBatch:
        text_only(plans, self)
        """Every level's tensors, built on the host: `forward` only launches."""
        slot_base, group = _bases(plans)
        has_children = {(p, s.parent) for p, plan in enumerate(plans)
                        for s in plan.segments if s.parent is not None}
        where: dict[tuple[int, int], tuple[int, int]] = {}  # kept node -> (start, n) in its pool
        nodes: dict[tuple[int, int], int] = {}              # kept node -> its index on its level
        levels = []
        for depth, level in enumerate(self._levels(plans)):
            offset, kept = 0, 0                             # the level's pool, laid out up front
            for p, i in level:
                if (p, i) in has_children:
                    where[(p, i)] = (offset, len(plans[p].segments[i].ids))
                    offset += len(plans[p].segments[i].ids)
                    nodes[(p, i)] = kept
                    kept += 1
            micro = [self._collate(plans, [level[j] for j in idx], depth, where, has_children,
                                   slot_base, nodes)
                     for idx in self._micro(plans, level)]
            levels.append((offset, kept, micro))
        return PrefixBatch(levels, group, sum(p.n_slots for p in plans))

    def _collate(self, plans, rows, depth, where, has_children, slot_base, nodes) -> PrefixMicroBatch:
        pad = self.model.pad_token_id
        segs = [plans[p].segments[i] for p, i in rows]
        width = max(len(s.ids) for s in segs)
        chains = [[(p, a) for a in self._ancestors(plans[p], i)] for p, i in rows]
        parents = kept_nodes = None
        if self._recurrent:
            if depth:
                parents = torch.tensor([nodes[(p, plans[p].segments[i].parent)] for p, i in rows],
                                       dtype=torch.long)
            kept = [(r, nodes[row]) for r, row in enumerate(rows) if row in has_children]
            if kept:
                kept_nodes = tuple(torch.tensor(t, dtype=torch.long) for t in zip(*kept))
        index, mask = [], [[] for _ in rows]
        for d in range(depth):                              # one ancestor per depth
            spans = [where[c[d]] for c in chains]
            w = max(n for _, n in spans)
            index.append(torch.tensor([list(range(start, start + n)) + [0] * (w - n)
                                       for start, n in spans], dtype=torch.long))
            for r, (_, n) in enumerate(spans):
                mask[r] += [1] * n + [0] * (w - n)
        ids, pos, kept, spans, local = [], [], ([], [], []), [], []
        for r, ((p, i), s) in enumerate(zip(rows, segs)):
            n = len(s.ids)
            left = width - n if self._recurrent and not depth and (p, i) in has_children else 0
            right = width - n - left
            ids.append([pad] * left + list(s.ids) + [pad] * right)
            pos.append([0] * left + list(s.pos) + [0] * right)
            mask[r] += [0] * left + [1] * n + [0] * right
            if (p, i) in has_children:
                start, _ = where[(p, i)]
                kept[0].extend([r] * n)
                kept[1].extend(range(left, left + n))
                kept[2].extend(range(start, start + n))
            if s.readout is not None:
                lo, hi = s.read_span()
                spans.append((r, left + lo, left + hi))
                local.append(slot_base[p] + s.readout[0])
        return PrefixMicroBatch(
            torch.tensor(ids, dtype=torch.long), torch.tensor(pos, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long), index,
            tuple(torch.tensor(t, dtype=torch.long) for t in kept) if kept[0] else None,
            Readout.from_spans(spans) if spans else None,
            torch.tensor(local, dtype=torch.long) if spans else None,
            parents, kept_nodes)

    def cost(self, plans: Sequence[ScoringPlan]) -> float:
        """Monotone in memory, in hidden-state tokens: the padded own tokens of
        every micro-batch (what per-layer checkpointing keeps per layer, like
        `CausalPerOptionExecutor.cost`) plus the pooled keys and values of the
        segments with children, weighted by their size against a hidden state."""
        cfg = self.model.backbone.config
        heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kv_weight = 2 * heads * head_dim / cfg.hidden_size
        parents = {(p, s.parent) for p, plan in enumerate(plans) for s in plan.segments
                   if s.parent is not None}
        kept = sum(len(plans[p].segments[i].ids) for p, i in parents)
        padded = 0
        for level in self._levels(plans):
            for idx in self._micro(plans, level):
                padded += len(idx) * max(len(plans[level[j][0]].segments[level[j][1]].ids)
                                         for j in idx)
        return padded + kv_weight * kept

    # ----------------------------------------------------------- forward
    def forward(self, batch: PrefixBatch) -> torch.Tensor:
        """One margin per slot, in slot order (the `Compiled.flat()` order)."""
        backbone, head = self.model.backbone, self.model.readout_head
        n_layers = backbone.config.num_hidden_layers
        if self._recurrent and torch.is_grad_enabled():
            raise RuntimeError("the prefix executor reads a recurrent backbone without gradients only")
        pools: list = []                                   # per depth: a `_Pool` or None
        scores, slots = [], []
        with _cache_through_checkpointing(backbone):
            for depth, (tokens, nodes, micro) in enumerate(batch.levels):
                pool = _Pool(n_layers, tokens, nodes) if tokens else None
                for mb in micro:
                    cache = _gather_cache(backbone.config, pools[:depth], mb.index, pool, mb.writes,
                                          mb.parents, mb.kept)
                    out = backbone(input_ids=mb.input_ids, position_ids=mb.position_ids,
                                   attention_mask=mb.attention_mask,
                                   past_key_values=cache, use_cache=False)
                    record = cache.record
                    record.open = False                    # a recompute reads, never writes
                    if record.calls != n_layers:
                        raise RuntimeError(
                            f"{record.calls} of {n_layers} layers read the ancestors' "
                            "keys: the backbone dropped past_key_values")
                    if mb.readout is not None:
                        scores.append(head(out.last_hidden_state, mb.readout))
                        slots.append(mb.slots)
                pools.append(pool)
        scores, slots = torch.cat(scores), torch.cat(slots)
        return scores[torch.argsort(slots)]

    def score(self, plans: Sequence[ScoringPlan]) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        margins = self.forward(self.pack(plans).to(device))
        return list(margins.split([p.n_slots for p in plans]))


def _gather_cache(config, pools, index, pool, writes, parents=None, kept=None):
    """A `DynamicCache` whose attention layers are `_GatherLayer`s: they read the
    ancestors from `pools` (per depth, a `_Pool`) at `index` (per depth: `(rows,
    width)` positions in that pool), and on the first forward write the tokens
    listed in `writes` -- `(rows, columns, pool positions)`, one entry per token
    -- into this level's `pool`. Its recurrent layers are `_ForkLayer`s: they start
    from the level above's states at `parents` and keep the `kept` rows' states."""
    from transformers import DynamicCache

    cache = DynamicCache(config=config)
    kinds = [type(layer).__name__ for layer in cache.layers]
    if set(kinds) - {"DynamicLayer", "LinearAttentionLayer"}:
        raise NotImplementedError(f"the prefix executor does not serve cache layers {set(kinds)}")
    # The layers report to a separate record, not to the cache: a cache <-> layer
    # cycle would keep the pools alive until the garbage collector runs.
    record = _Record(len(cache.layers), pool, writes, kept)
    gather, fork = _gather_layer_class(), _fork_layer_class()
    parent = pools[-1] if pools else None
    cache.layers = [fork(record, i, parent, parents) if kind == "LinearAttentionLayer" else
                    gather(record, i, [(a.keys[i], a.values[i]) for a in pools], index)
                    for i, kind in enumerate(kinds)]
    cache.record = record
    return cache


class _Pool:
    """A level's kept keys and values, per layer `(heads, tokens, head_dim)`.
    Allocated on the first write and filled in place by the level's forward:
    each segment with children is stored once, with nothing in between. A
    micro-batch writes all its kept tokens with one gather and one
    `index_copy_` per layer, however many rows it keeps. A recurrent layer keeps,
    in their place, one recurrent and one convolution state per segment."""

    def __init__(self, n_layers: int, tokens: int, nodes: int = 0):
        self.tokens = tokens
        self.nodes = nodes
        self.keys: list = [None] * n_layers
        self.values: list = [None] * n_layers
        self.recurrent: list = [None] * n_layers
        self.conv: list = [None] * n_layers

    def write(self, layer: int, writes, keys, values):
        if self.keys[layer] is None:
            shape = (keys.shape[1], self.tokens, keys.shape[3])
            self.keys[layer], self.values[layer] = keys.new_empty(shape), values.new_empty(shape)
        self.keys[layer] = _KeepTokens.apply(self.keys[layer], keys, *writes)
        self.values[layer] = _KeepTokens.apply(self.values[layer], values, *writes)

    def write_state(self, layer: int, kept, recurrent, conv):
        rows, at = kept
        if self.recurrent[layer] is None:
            self.recurrent[layer] = recurrent.new_empty((self.nodes, *recurrent.shape[1:]))
            self.conv[layer] = conv.new_empty((self.nodes, *conv.shape[1:]))
        self.recurrent[layer][at] = recurrent[rows]
        self.conv[layer][at] = conv[rows]


class _KeepTokens(torch.autograd.Function):
    """`pool[:, at] = states[rows, :, cols]` in place: states `(b, heads, w, d)`,
    pool `(heads, tokens, d)`, one entry of `rows` / `cols` / `at` per token.

    The indices live on `ctx`, not in `save_for_backward`: the write runs in
    the first forward only, and a checkpoint recompute that skips it must save
    the same tensors as the forward did (built-in indexing would save them)."""

    @staticmethod
    def forward(ctx, pool, states, rows, cols, at):
        ctx.rows, ctx.cols, ctx.at, ctx.shape = rows, cols, at, states.shape
        pool.index_copy_(1, at, states[rows, :, cols].transpose(0, 1))
        ctx.mark_dirty(pool)
        return pool

    @staticmethod
    def backward(ctx, grad):
        grad_pool = grad.index_fill(1, ctx.at, 0) if ctx.needs_input_grad[0] else None
        grad_states = None
        if ctx.needs_input_grad[1]:
            grad_states = grad.new_zeros(ctx.shape)
            grad_states[ctx.rows, :, ctx.cols] = grad.index_select(1, ctx.at).transpose(0, 1)
        return grad_pool, grad_states, None, None, None


class _Record:
    """One prefix cache's bookkeeping: which layers ran their first forward (and
    wrote their rows into the level's pool) -- `open` until the forward ends, so
    a checkpoint recompute in backward only reads."""

    def __init__(self, n_layers: int, pool, writes, kept=None):
        self.pool, self.writes, self.kept = pool, writes, kept
        self.seen = [False] * n_layers
        self.calls = 0
        self.open = True


@lru_cache(maxsize=None)
def _gather_layer_class():
    """`_GatherLayer`, built on first use so that importing this module does not
    import transformers."""
    from transformers.cache_utils import DynamicLayer

    class _GatherLayer(DynamicLayer):
        """One layer of the prefix cache. Stores nothing between calls: `update`
        gathers the rows' ancestors from the pools and appends the new keys and
        values, so a forward and its checkpoint recompute see the same keys. On
        the first forward it also writes the kept rows' new keys and values into
        the level's pool (for the children)."""

        def __init__(self, record, layer: int, pools, index):
            super().__init__()
            self.record, self.layer, self.pools, self.index = record, layer, pools, index
            if pools:
                k0 = pools[0][0]
                rows, width = index[0].shape[0], sum(ix.shape[1] for ix in index)
                stub = k0.new_zeros(()).expand(rows, k0.shape[0], width, k0.shape[2])
                self.keys, self.values = stub, stub             # shape only: the mask's lengths
                self.dtype, self.device = k0.dtype, k0.device
                self.is_initialized = True

        def update(self, key_states, value_states, *args, **kwargs):
            record = self.record
            if record.open and not record.seen[self.layer]:
                record.seen[self.layer] = True
                record.calls += 1
                if record.writes is not None:
                    record.pool.write(self.layer, record.writes, key_states, value_states)
            if not self.pools:
                return key_states, value_states
            keys = [k[:, ix].transpose(0, 1) for (k, _), ix in zip(self.pools, self.index)]
            values = [v[:, ix].transpose(0, 1) for (_, v), ix in zip(self.pools, self.index)]
            return torch.cat(keys + [key_states], -2), torch.cat(values + [value_states], -2)

    return _GatherLayer


class _Forked:
    """A recurrent layer's starting states, copied per row only when the layer
    reads them, so that one layer's copies are gone before the next one's."""

    def __init__(self, states, parents):
        self.states, self.parents = states, parents

    def __getitem__(self, state_idx):
        return self.states[self.parents]


@lru_cache(maxsize=None)
def _fork_layer_class():
    """`_ForkLayer`, built on first use like `_GatherLayer`."""
    from transformers.cache_utils import LinearAttentionLayer

    class _ForkLayer(LinearAttentionLayer):
        """One recurrent layer of the prefix cache: its rows start where their
        parents ended, and the kept rows' final states go to the level's pool."""

        def __init__(self, record, layer: int, parent, parents):
            super().__init__()
            self.record, self.layer = record, layer
            self.tail = None
            if parent is not None:
                self.recurrent_states = _Forked(parent.recurrent[layer], parents)
                self.conv_states[0] = self.tail = parent.conv[layer][parents]
                self.has_previous_state[0] = True

        def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
            past = self.conv_states[0]
            full = conv_states if past is None else torch.cat([past, conv_states], -1)
            short = max(conv_kernel_size - full.shape[-1], 0)     # the left context is zeros
            self.tail = torch.nn.functional.pad(full, (short, 0))[..., -conv_kernel_size:]
            return full

        def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
            record = self.record
            record.calls += 1
            if record.kept is not None:
                record.pool.write_state(self.layer, record.kept, recurrent_states, self.tail)
            return recurrent_states

    return _ForkLayer


@contextmanager
def _cache_through_checkpointing(module):
    """Let transformers' checkpointed layers keep `past_key_values` (our cache
    layers are pure). Restores each layer's own setting."""
    layers = [m for m in module.modules() if hasattr(type(m), "_can_checkpoint_with_cache")]
    saved = [m.__dict__.get("_can_checkpoint_with_cache") for m in layers]
    for m in layers:
        m._can_checkpoint_with_cache = True
    try:
        yield
    finally:
        for m, value in zip(layers, saved):
            if value is None:
                del m._can_checkpoint_with_cache
            else:
                m._can_checkpoint_with_cache = value
