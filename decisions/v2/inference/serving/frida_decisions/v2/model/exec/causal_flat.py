"""`CausalFlatExecutor`: the segment tree in one pass through a hybrid's layers.

Same visibility and margins as `exec.causal`'s executors (an option sees the
state, its question and itself, causally), but one pass through the layers, as
FRIDA-Decisions' packing does, instead of one pass per level of the tree.

Every token of the batch lives once in a flat `(1, N + 1, d)` tensor, the last
token being empty (exactly zero, it stays zero through the layers). Norms,
projections and MLPs run on the flat tensor, one launch each however the tree
looks. Only the two mixers look at the tree:

* Attention: the states attend to themselves causally, and every other token
  attends through one tree mask per question (`_Attend`).
* Gated DeltaNet: the convolution runs once over the flat tokens, each segment
  preceded by the tail of its ancestors. The recurrent kernel, which cannot skip
  anything, runs in waves (`_Wave`). A segment whose repeats would cost more than
  a wave (`FORK_TOKENS`) gets a wave of its own and forks its state; any other
  travels in its children's rows and is read again per child by the recurrent
  kernel alone, its output taken once. So a long state takes one wave and
  `question + option` rows the next, and a short state with its options takes one.

On a GPU (`CausalFlatEagerExecutor` keeps PyTorch's way): norms take one Triton launch each and a
short wave one (`causal_kernels`), a block's projections one matmul. A bf16 output may round one
step differently.

Images. Only a state carries them, and a state's tokens are flat tokens like any
other: the tower runs once per pass over all the pass's images (`_Images`, each
image once, whatever hangs below it), its features take the place of the
`<|image_pad|>` embeddings, and the layers see no difference. What they need is the
state's M-RoPE positions, so a pass with images has `(3, N + 1)` positions; text
tokens carry one position on all three axes, as under per_option.

Rows of a wave. With the `fla` kernels on a GPU they run concatenated, without
pads (`cu_seqlens`): the chunked kernel takes few launches and is the faster one on
much data, the per-token kernel takes one launch and is the faster one on little
(`_kernel`). `CausalFlatPaddedExecutor` and the reference PyTorch kernel pad to the
longest row instead: the empty token has a zero key, value and decay, so the state
passes through it unchanged, also in rows whose final state is kept.

Inference only: a backbone is trained under `per_option` or `prefix`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from typing import Sequence

import torch
import torch.nn.functional as F

from frida_decisions.v2.model.exec.causal import CONFIG, _bases, _micro_groups, _padded_groups
from frida_decisions.v2.model.head import Readout
from frida_decisions.v2.model.layout import ScoringPlan

# Tokens a segment may repeat in the recurrent kernel before a wave of its own is cheaper.
FORK_TOKENS = 1024
# Use the per-token kernel below both limits.
PER_TOKEN_TOKENS = 4096
PER_TOKEN_ROW = 1024
# PER_TOKEN_TOKENS is an H100's (132 SMs, many rows side by side). The per-token kernel takes a
# row's tokens one after another, so on a card with fewer SMs the chunked kernel pays off sooner:
# the budget scales with the SM count.
H100_SMS = 132


@lru_cache(maxsize=None)
def _sm_share() -> float:
    if not torch.cuda.is_available():
        return 1.0
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return min(1.0, sms / H100_SMS)


def _per_token_tokens() -> int:
    return int(PER_TOKEN_TOKENS * _sm_share())


@dataclass
class _Wave:
    """One call of the recurrent kernel. Rows of `level` start from the states of
    `level - 1`, `parents[row]`; `at[t]` takes the output `(row[t], col[t])`, and
    `kept` holds `(row, node)` per state that the next level starts from."""
    level: int
    tokens: torch.Tensor                       # (b, w) flat positions, or (1, T) concatenated
    parents: torch.Tensor | None
    row: torch.Tensor
    col: torch.Tensor
    at: torch.Tensor
    kept: tuple[torch.Tensor, torch.Tensor] | None
    cu: torch.Tensor | None = None             # (b + 1,) row starts when concatenated
    longest: int = 0
    conv: torch.Tensor | None = None           # flat `tokens`, as positions in the convolution's output
    writes: torch.Tensor | None = None         # flat `tokens`' places in the mixer's output, -1 if read again
    chunks: torch.Tensor | None = None         # FLA's chunk indices: from the GPU they cost a sync per layer

    def to(self, device):
        return _Wave(self.level, self.tokens.to(device),
                     None if self.parents is None else self.parents.to(device),
                     self.row.to(device), self.col.to(device), self.at.to(device),
                     None if self.kept is None else tuple(t.to(device) for t in self.kept),
                     None if self.cu is None else self.cu.to(device), self.longest,
                     None if self.conv is None else self.conv.to(device),
                     None if self.writes is None else self.writes.to(device),
                     None if self.chunks is None else self.chunks.to(device))


@dataclass
class _Attend:
    """One attention call: `queries` `(b, w)` over `keys` `(b, W)`. No keys: the
    queries attend to themselves causally. `at[t]` takes the output `(row[t], col[t])`."""
    queries: torch.Tensor
    keys: torch.Tensor | None
    mask: torch.Tensor | None                  # (b, 1, w, W) bool
    row: torch.Tensor
    col: torch.Tensor
    at: torch.Tensor

    contiguous_start: int | None = None
    suffix_masks: list[tuple[int, torch.Tensor]] | None = None
    kv_span: tuple[int, int] | None = None  # state length and maximum row length in a single tree

    def materialize_mask(self, device):
        if self.suffix_masks is None:
            return None if self.mask is None else self.mask.to(device)
        if len(self.suffix_masks) == 1:
            prefix, tail = self.suffix_masks[0]
            return F.pad(tail.to(device), (prefix, 0), value=True)[None, None]
        rows, width = self.queries.shape
        mask = torch.zeros((rows, 1, width, self.keys.shape[1]), dtype=torch.bool, device=device)
        for i, (prefix, tail) in enumerate(self.suffix_masks):
            q, k = tail.shape
            mask[i, 0, :q, :prefix] = True
            mask[i, 0, :q, prefix:prefix + k] = tail.to(device)
            mask[i, 0, q:, 0] = True
        return mask

    def to(self, device):
        return _Attend(self.queries.to(device), None if self.keys is None else self.keys.to(device),
                       self.materialize_mask(device),
                       self.row.to(device), self.col.to(device), self.at.to(device), self.contiguous_start, kv_span=self.kv_span)


@dataclass
class _Images:
    """A pass's images, the tower's input in request order, and the flat position of
    every image token, whose embeddings their features replace in that order. Pixels stay
    per request up to the forward: joining them on the host costs more than moving them."""
    pixel_values: list[torch.Tensor]
    image_grid_thw: torch.Tensor               # (images, 3)
    at: torch.Tensor                           # (image tokens,)

    rgb8_tables: list[torch.Tensor | None] | None = None
    host_grid: torch.Tensor | None = None

    def to(self, device):
        pixels = [p.to(device) for p in self.pixel_values]
        if self.rgb8_tables is not None:
            for i, table in enumerate(self.rgb8_tables):
                if table is None:
                    continue
                values = pixels[i]
                if values.is_cuda:
                    from frida_decisions.v2.model.exec.rgb8_patches import expand_rgb8
                    pixels[i] = expand_rgb8(values, table.to(device))
                else:
                    bits = table.to(device)[values.long()].view(-1, 3, 1, 256)
                    pixels[i] = bits.expand(-1, -1, 2, -1).reshape(-1, 1536).view(torch.bfloat16)
        host_grid = self.image_grid_thw if self.image_grid_thw.device.type == "cpu" else self.host_grid
        return _Images(pixels, self.image_grid_thw.to(device), self.at.to(device), host_grid=host_grid)


@dataclass
class _Pass:
    """One pass through the layers, for some of the batch's requests."""
    ids: torch.Tensor                          # (N,)
    positions: torch.Tensor                    # (N + 1,), or (3, N + 1) with images
    decays: torch.Tensor                       # (N + 1, 1): 0 at the empty token
    conv_in: torch.Tensor                      # (C,) flat positions, with the ancestors' tails
    waves: list[_Wave]
    nodes: list[int]                           # kept states per level
    attends: list[_Attend]
    readout: Readout
    images: _Images | None = None
    skip_final_root: bool = False

    def to(self, device):
        return _Pass(self.ids.to(device), self.positions.to(device), self.decays.to(device),
                     self.conv_in.to(device),
                     [w.to(device) for w in self.waves], self.nodes,
                     [a.to(device) for a in self.attends], self.readout.to(device),
                     None if self.images is None else self.images.to(device), self.skip_final_root)


@dataclass
class FlatBatch:
    passes: list[_Pass]
    group: torch.Tensor
    n_slots: int

    def to(self, device):
        return FlatBatch([p.to(device) for p in self.passes], self.group.to(device), self.n_slots)

    @property
    def padded_tokens(self) -> int:
        return sum(int(p.ids.numel()) for p in self.passes)


def _budgeted(lengths: list[int], budget: int | None) -> list[list[int]]:
    """Indices per wave of concatenated rows, up to the budget in tokens."""
    if budget is None:
        return [list(range(len(lengths)))]
    groups, current, tokens = [], [], 0
    for i, n in enumerate(lengths):
        if current and tokens + n > budget:
            groups.append(current)
            current, tokens = [], 0
        current.append(i)
        tokens += n
    groups.append(current)
    return groups


def _padded(rows: list[list[int]], pad: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [pad] * (width - len(r)) for r in rows], dtype=torch.long)


@lru_cache(maxsize=1)
def _verified_grouped_bf16_build():
    # This path follows the private grouped-linear API and the installed ATen BF16 reduction tree.
    from transformers import __version__ as transformers_version

    return torch.__version__.split("+")[0] == "2.13.0" and transformers_version == "5.19.0"


@lru_cache(maxsize=None)
def _fla_kernels():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    from frida_decisions.v2.model.exec.causal_kernels import pin_fla_kernels

    pin_fla_kernels()
    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def _chunk_indices(starts: torch.Tensor) -> torch.Tensor:
    from fla.ops.utils.index import prepare_chunk_indices

    return prepare_chunk_indices(starts, 64)


@lru_cache(maxsize=None)
def _norm_kernels():
    from frida_decisions.v2.model.exec.causal_kernels import recurrent_wave, rms_norm, rms_norm_gated

    return rms_norm, rms_norm_gated, recurrent_wave


class CausalFlatExecutor:
    """See the module doc. `model` is what the other causal executors take."""
    trains_with_checkpointing = False
    concatenates = True
    fuses_norms = True
    fp8_experts = False

    def __init__(self, model, max_batch_tokens=CONFIG):
        self.model = model
        self._budget = max_batch_tokens
        mode = os.getenv("FDD_FLAT_TEXT_PACK")
        self.text_pack = mode in {"1", "2", "fast", "parallel"}
        self.fast_text = mode in {"fast", "parallel"}
        self.parallel_attention = mode == "parallel"
        self.compact_readout = mode == "2"
        # What a norm or a mixer computes from its weights alone. On the model, which outlives
        # an executor. Inference only: weights are taken as fixed after the first forward.
        self._cached = model.__dict__.setdefault("_flat_from_weights", {})

    @property
    def max_batch_tokens(self) -> int | None:
        if self._budget is CONFIG:
            return self.model.config.max_batch_tokens
        return self._budget

    @property
    def _stack(self):
        """The module that owns the layers (through a PEFT wrapper too)."""
        backbone = self.model.backbone
        return backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone

    # ------------------------------------------------------------ layout
    def pack(self, plans: Sequence[ScoringPlan]) -> FlatBatch:
        """Everything on the host, `forward` only launches. A pass holds whole
        requests up to the budget in tokens (a larger request gets its own)."""
        budget = self.max_batch_tokens
        passes, current, tokens = [], [], 0
        for plan in plans:
            n = sum(len(s.ids) for s in plan.segments)
            if current and budget is not None and tokens + n > budget:
                passes.append(self._layout(current))
                current, tokens = [], 0
            current.append(plan)
            tokens += n
        if current:
            passes.append(self._layout(current))
        _, group = _bases(plans)
        return FlatBatch(passes, group, sum(p.n_slots for p in plans))

    def _layout(self, plans: list[ScoringPlan]) -> _Pass:
        ids, positions, spans = [], [], []
        where = []                                 # per plan, per segment: flat (start, end)
        m_rope = []                                # (first flat token, its state's (3, n) positions)
        for plan in plans:
            ranges = []
            for s in plan.segments:
                ranges.append((len(ids), len(ids) + len(s.ids)))
                ids += list(s.ids)
                if torch.is_tensor(s.pos):         # a state with images
                    m_rope.append((len(positions), s.pos))
                    positions += [0] * len(s.ids)
                else:
                    positions += list(s.pos)
            where.append(ranges)
            for leaf in plan.leaves():
                lo, hi = plan.segments[leaf].read_span()
                start = ranges[leaf][0]
                spans.append((0, start + lo, start + hi))
        n = len(ids)
        children = [[[] for _ in plan.segments] for plan in plans]
        for p, plan in enumerate(plans):
            for i, s in enumerate(plan.segments):
                if s.parent is not None:
                    children[p][s.parent].append(i)
        waves, nodes = self._waves(plans, where, children, n)
        conv_in, conv_out = self._conv(plans, where, n)
        for wave in waves:
            wave.conv = conv_out[wave.tokens.view(-1)]
            if wave.cu is not None:
                wave.writes = torch.full((wave.tokens.numel(),), -1, dtype=torch.long)
                wave.writes[wave.col] = wave.at
                if wave.tokens.numel() > _per_token_tokens() or wave.longest > PER_TOKEN_ROW:
                    wave.chunks = _chunk_indices(wave.cu)
        decays = torch.ones(n + 1, 1)
        decays[n] = 0
        flat_positions = torch.tensor(positions + [0], dtype=torch.long)
        if m_rope:
            flat_positions = flat_positions.expand(3, -1).clone()
            for start, pos in m_rope:
                flat_positions[:, start:start + pos.shape[1]] = pos
        return _Pass(torch.tensor(ids, dtype=torch.long), flat_positions, decays,
                     conv_in, waves, nodes,
                     self._attends(plans, where, children, n), Readout.from_spans(spans),
                     self._images(plans, where),
                     self.text_pack and all((not p.mm or self.parallel_attention) and p.root.readout is None for p in plans))

    def _images(self, plans, where) -> _Images | None:
        """Every image of the pass, once: a plan's images are its state's."""
        with_images = [(p, plan) for p, plan in enumerate(plans) if plan.mm]
        if not with_images:
            return None
        token = self.model.vision.image_token_id
        at = [where[p][0][0] + i for p, plan in with_images
              for i, t in enumerate(plan.root.ids) if t == token]
        tables = [plan.mm.get("rgb8_table") for _, plan in with_images]
        return _Images([plan.mm["pixel_values"] for _, plan in with_images],
                       torch.cat([plan.mm["image_grid_thw"] for _, plan in with_images]),
                       torch.tensor(at, dtype=torch.long), tables if any(t is not None for t in tables) else None)

    def _conv(self, plans, where, n):
        """Each segment after the tail of its ancestors. The first entry is the
        empty token: its output is the empty token's, and zero."""
        kernel = getattr(self._stack.config, "linear_conv_kernel_dim", 1)
        if self.parallel_attention and n >= 2048:
            import numpy as np

            keep = kernel - 1
            count = sum(len(plan.segments) for plan in plans)
            index = np.arange(n + 1, dtype=np.int64)
            sequence = np.empty(1 + n + count * keep, dtype=np.int64)
            read = np.empty_like(index)
            sequence[0], read[n] = n, 0
            cursor = 1
            for plan, ranges in zip(plans, where, strict=True):
                tails = {}
                for i, (segment, (start, end)) in enumerate(zip(plan.segments, ranges, strict=True)):
                    past = () if segment.parent is None else tails[segment.parent]
                    prefix = (n,) * (keep - len(past)) + past
                    sequence[cursor:cursor + keep] = prefix
                    cursor += keep
                    size = end - start
                    sequence[cursor:cursor + size] = index[start:end]
                    read[start:end] = index[:size] + cursor
                    cursor += size
                    tails[i] = (past + tuple(range(max(start, end - keep), end)))[-keep:] if keep else ()
            return torch.from_numpy(sequence), torch.from_numpy(read)
        sequence, read = [n], [0] * (n + 1)
        for p, plan in enumerate(plans):
            tails = {}
            for i, s in enumerate(plan.segments):
                start, end = where[p][i]
                past = [] if s.parent is None else tails[s.parent]
                keep = kernel - 1
                sequence += ([n] * keep + past)[-keep:] if keep else []
                for t in range(start, end):
                    read[t] = len(sequence)
                    sequence.append(t)
                tails[i] = (past + list(range(start, end)))[-keep:] if keep else []
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(read, dtype=torch.long)

    def _waves(self, plans, where, children, n):
        """Per level, rows `(tokens, parent state, takes, kept state)`; a take is
        `(first flat token, first column, size)`. Level 0 starts from zero."""
        rows: dict[int, list] = {}
        counts: dict[int, int] = {}
        concatenate = self.concatenates and find_spec("fla") is not None and \
            next(self.model.parameters()).device.type == "cuda"
        for p, plan in enumerate(plans):
            kids = children[p]
            # How many rows would read each segment, children before parents.
            reads, forking = [1] * len(kids), set()
            for i in reversed(range(len(kids))):
                if kids[i]:
                    reads[i] = sum(1 if c in forking else reads[c] for c in kids[i])
                    if len(plan.segments[i].ids) * (reads[i] - 1) > FORK_TOKENS:
                        forking.add(i)

            def forks(i):
                return i in forking

            level, node, base = {}, {}, {}
            done = set()
            for i, s in enumerate(plan.segments):
                j = s.parent
                base[i] = None if j is None else j if forks(j) else base[j]
                here = 0 if base[i] is None else level[base[i]] + 1
                keep = forks(i)
                if keep:
                    level[i] = here
                    node[i] = counts.get(here, 0)
                    counts[here] = node[i] + 1
                if kids[i] and not keep:
                    continue                       # travels with its children
                path, a = [], i
                while a != base[i]:
                    path.append(a)
                    a = plan.segments[a].parent
                tokens, takes = [], []
                for a in reversed(path):
                    first, last = where[p][a]
                    if a not in done:
                        done.add(a)
                        takes.append((first, len(tokens), last - first))
                    tokens += list(range(first, last))
                rows.setdefault(here, []).append((tokens, None if base[i] is None else node[base[i]],
                                                  takes, node.get(i)))
        waves = []
        for depth in sorted(rows):
            members = rows[depth]
            lengths = [len(r[0]) for r in members]
            groups = _budgeted(lengths, self.max_batch_tokens) if concatenate else \
                _padded_groups(lengths, self.max_batch_tokens)
            for group in groups:
                waves.append(self._wave(depth, [members[g] for g in group], n, concatenate))
        return waves, [counts.get(depth, 0) for depth in range(len(rows))]

    @staticmethod
    def _wave(level, members, n, concatenate):
        if concatenate and len(members) == 1:
            tokens, parent, takes, node = members[0]
            size = len(tokens)
            if size >= 2048 and len(takes) == 1 and takes[0][1:] == (0, size):
                first = takes[0][0]
                cols = torch.arange(size)
                at = cols if first == 0 else torch.arange(first, first + size)
                parents = torch.tensor([parent], dtype=torch.long) if level else None
                kept = (torch.tensor([0]), torch.tensor([node])) if node is not None else None
                return _Wave(level, at.view(1, -1), parents, torch.zeros_like(cols), cols, at, kept,
                             torch.tensor([0, size], dtype=torch.long), size)
        rows, cols, at, kept = [], [], [], []
        offset = 0
        for r, (tokens, _, takes, node) in enumerate(members):
            for flat, column, size in takes:
                rows += [0 if concatenate else r] * size
                cols += list(range(offset + column, offset + column + size))
                at += list(range(flat, flat + size))
            if node is not None:
                kept.append((r, node))
            if concatenate:
                offset += len(tokens)
        parents = None if not level else torch.tensor([m[1] for m in members], dtype=torch.long)
        lengths = [len(m[0]) for m in members]
        tokens = torch.tensor([[t for m in members for t in m[0]]], dtype=torch.long) if concatenate \
            else _padded([m[0] for m in members], n)
        cu = torch.tensor([0] + lengths, dtype=torch.long).cumsum(0) if concatenate else None
        return _Wave(level, tokens, parents,
                     torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long),
                     torch.tensor(at, dtype=torch.long),
                     tuple(torch.tensor(t, dtype=torch.long) for t in zip(*kept)) if kept else None,
                     cu, max(lengths))

    def _attends(self, plans, where, children, n):
        c = self.model.config
        cap_options = c.max_options_per_row or 10**12
        cap_tokens = c.max_row_tokens if c.max_row_tokens is not None else 10**12
        roots = [range(*where[p][0]) for p in range(len(plans))]
        out = []
        for group in _micro_groups([len(r) for r in roots], self.max_batch_tokens):
            members = [roots[g] for g in group]
            if self.parallel_attention and len(members) == 1:
                root = members[0]
                at = torch.arange(root.start, root.stop)
                cols = torch.arange(len(root)) if root.start else at
                out.append(_Attend(at.view(1, -1), None, None, torch.zeros_like(at), cols, at, root.start))
                continue
            members = [list(member) for member in members]
            rows = [r for r, m in enumerate(members) for _ in m]
            cols = [c for m in members for c in range(len(m))]
            at = [t for m in members for t in m]
            out.append(_Attend(_padded(members, n), None, None, torch.tensor(rows, dtype=torch.long),
                               torch.tensor(cols, dtype=torch.long), torch.tensor(at, dtype=torch.long),
                               members[0][0] if len(members) == 1 and members[0] else None))
        below = []
        prefixes = []
        defer = self.fast_text and all((not plan.mm or self.parallel_attention) and len(plan.root.ids) >= 2048
                                       for plan in plans)
        for p, plan in enumerate(plans):
            for top in children[p][0]:
                prefix = len(plan.root.ids) if defer else 0
                rows = self._rows_below(plan, where[p], children[p], top, cap_options, cap_tokens, prefix)
                below.extend(rows)
                if defer:
                    prefixes.extend([prefix] * len(rows))
        if below:
            compact = self.text_pack and not any(plan.mm for plan in plans)
            for group in _micro_groups([len(r[0]) for r in below], self.max_batch_tokens):
                out.append(self._attend([below[g] for g in group], n, compact,
                                        [prefixes[g] for g in group] if defer else None))
        if defer and len(plans) == 1:
            # Entry zero is the contiguous root. Single-row suffix calls read the
            # same state tokens in the same order, including split option rows.
            singleton = [a for a in out[1:] if a.keys.shape[0] == 1]
            if singleton:
                span = (len(plans[0].root.ids), max(a.keys.shape[1] for a in singleton))
                out[0].kv_span = span
                for a in singleton:
                    a.kv_span = span
        return out

    @staticmethod
    def _rows_below(plan, ranges, kids, top, cap_options, cap_tokens, prefix=0):
        """Rows `(keys, first query column, allowed)` of one question: the state,
        the question and as many options as the row may hold. The question is
        asked in its first row, and is only read by the later ones."""
        inner, leaves, stack = [], [], [top]
        while stack:
            i = stack.pop()
            (inner if kids[i] else leaves).append(i)
            stack.extend(reversed(kids[i]))
        lists, current, tokens = [], [], 0
        size = sum(len(plan.segments[i].ids) for i in inner)
        for leaf in leaves:
            n = len(plan.segments[leaf].ids)
            if current and (len(current) >= cap_options or size + tokens + n > cap_tokens):
                lists.append(current)
                current, tokens = [], 0
            current.append(leaf)
            tokens += n
        if current:
            lists.append(current)
        rows = []
        state = [] if prefix else list(range(*ranges[0]))
        root_tokens = torch.arange(*ranges[0]) if prefix else None
        for r, leafs in enumerate(lists or [[]]):
            keys, spans = list(state), {0: (0, len(state))}
            for i in inner + leafs:
                spans[i] = (len(keys), len(keys) + len(plan.segments[i].ids))
                keys += list(range(*ranges[i]))
            offset = len(state) + (0 if r == 0 else size)
            allowed = torch.zeros((len(keys) - offset, len(keys)), dtype=torch.bool)
            for i in (inner if r == 0 else []) + leafs:
                a0, a1 = spans[i]
                anc = plan.segments[i].parent
                while anc is not None and not (prefix and anc == 0):
                    allowed[a0 - offset:a1 - offset, spans[anc][0]:spans[anc][1]] = True
                    anc = plan.segments[anc].parent
                allowed[a0 - offset:a1 - offset, a0:a1] = torch.ones(
                    (a1 - a0, a1 - a0), dtype=torch.bool).tril()
            if root_tokens is not None:
                keys = torch.cat((root_tokens, torch.tensor(keys, dtype=torch.long)))
                offset += prefix
            rows.append((keys, offset, allowed))
        return rows

    @staticmethod
    def _attend(rows, n, compact=False, prefixes=None) -> _Attend:
        queries = [r[0][r[1]:] for r in rows]
        keys = [r[0] for r in rows]
        width, full = max(map(len, queries)), max(map(len, keys))
        tails = None
        if prefixes is not None:
            tails = [(prefix, row[2]) for prefix, row in zip(prefixes, rows, strict=True)]
            if len(rows) == 1:
                q = queries[0]
                return _Attend(q.view(1, -1), keys[0].view(1, -1), None,
                               torch.zeros_like(q), torch.arange(q.numel()), q, suffix_masks=tails)
            pad = torch.nn.utils.rnn.pad_sequence
            at = torch.cat(queries)
            row_ids = torch.cat([torch.full_like(q, i) for i, q in enumerate(queries)])
            columns = torch.cat([torch.arange(q.numel()) for q in queries])
            return _Attend(pad(queries, batch_first=True, padding_value=n),
                           pad(keys, batch_first=True, padding_value=n), None, row_ids, columns, at,
                           suffix_masks=tails)
        elif compact and len(rows) == 1:
            mask = rows[0][2][None, None]
        else:
            mask = torch.zeros((len(rows), 1, width, full), dtype=torch.bool)
            mask[:, 0, :, 0] = True                    # a pad must attend to something; nothing reads it
            for r, (row, offset, allowed) in enumerate(rows):
                w = len(row) - offset
                mask[r, 0, :w, :] = False
                mask[r, 0, :w, :len(row)] = allowed
        key_tensor = _padded(keys, n)
        return _Attend(_padded(queries, n), key_tensor, mask,
                       torch.tensor([r for r, q in enumerate(queries) for _ in q], dtype=torch.long),
                       torch.tensor([c for q in queries for c in range(len(q))], dtype=torch.long),
                       torch.tensor([t for q in queries for t in q], dtype=torch.long), suffix_masks=tails)

    def cost(self, plans: Sequence[ScoringPlan]) -> int:
        """The batch's tokens: every one of them is computed once."""
        return sum(len(s.ids) for plan in plans for s in plan.segments)

    # ----------------------------------------------------------- forward
    def forward(self, batch: FlatBatch) -> torch.Tensor:
        """One margin per slot, in slot order (the `Compiled.flat()` order)."""
        if torch.is_grad_enabled():
            raise RuntimeError("the flat executor reads a backbone without gradients only")
        return torch.cat([self._forward(p) for p in batch.passes])

    def _from_weights(self, module, build, x: torch.Tensor):
        value = self._cached.get(module)
        if value is None or (value[0] if isinstance(value, tuple) else value).device != x.device:
            value = self._cached[module] = build()
        return value

    def _norm(self, norm, x: torch.Tensor) -> torch.Tensor:
        if not self._fused:
            return norm(x)
        scale = self._from_weights(norm, lambda: 1.0 + norm.weight.float(), x)
        return _norm_kernels()[0](x, scale, norm.eps)

    def _project(self, modules: tuple, x: torch.Tensor):
        """What `modules` make of `x`, in one launch where they are plain linear layers on a GPU:
        one weight over all of them, and theirs become views of it, so nothing is stored twice."""
        if not self._fused or not all(type(m) is torch.nn.Linear and m.bias is None for m in modules):
            self._whole = None
            return tuple(m(x) for m in modules)

        def merge():
            weight = torch.cat([m.weight.data for m in modules])
            sizes = [m.weight.shape[0] for m in modules]
            for m, part in zip(modules, weight.split(sizes)):
                m.weight.data = part
            return weight, sizes

        weight, sizes = self._from_weights(modules, merge, x)
        self._whole = F.linear(x, weight)          # the kernel of a wave reads it in place
        return self._whole.split(sizes, -1)

    def _sparse_block_fp8(self, block, x: torch.Tensor) -> torch.Tensor:
        """`Qwen3_5MoeSparseMoeBlock` with its experts in block FP8 on vLLM's kernel (`fp8_experts`)."""
        from frida_decisions.v2.model.exec.fp8_experts import block_fp8_weights, vllm_experts
        h = x.view(-1, x.shape[-1])
        shared = block.shared_expert(h)
        _, weights, chosen = block.gate(h)
        stored = self._fp8_weights(block.experts, block_fp8_weights, x)
        out = vllm_experts(stored, h, chosen, weights)
        return (out + torch.sigmoid(block.shared_expert_gate(h)) * shared).view(x.shape)

    def _fp8_weights(self, experts, build, x):
        """Built once on the device the model runs on: the build drops the bf16 weights."""
        value = self._cached.get(("fp8", experts))
        if value is None:
            value = self._cached[("fp8", experts)] = build(experts)
        elif value["gate_up_proj"].device != x.device:
            raise RuntimeError("FP8 experts were built on another device; load the model where it runs")
        return value

    def _check_fp8(self, experts) -> None:
        """FP8 or a loud failure, never bf16 under the FP8 name: a PEFT wrapper hides the fused
        expert weights (merge the adapter first), and the kernel needs a GPU with Triton."""
        if not hasattr(experts, "gate_up_proj"):
            raise RuntimeError(f"flat_fp8: {type(experts).__name__} has no fused expert weights; "
                               "merge the adapter before quantizing")
        if not self._fused:
            raise RuntimeError("flat_fp8 needs CUDA and Triton (vLLM's fused MoE)")

    def quantize_experts(self) -> None:
        """Every MoE layer's experts in FP8 now, so that the first request and a timing do
        not pay for it. A layer at a time either way: the peak is one layer's FP8 copy."""
        if not self.fp8_experts:
            return
        from frida_decisions.v2.model.exec.fp8_experts import block_fp8_weights
        for layer in self._stack.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None:
                if not hasattr(experts, "gate_up_proj"):
                    raise RuntimeError(f"flat_fp8: {type(experts).__name__} has no fused expert weights; "
                                       "merge the adapter before quantizing")
                self._fp8_weights(experts, block_fp8_weights, experts.gate_up_proj)
                torch.cuda.empty_cache()

    def _mlp(self, mlp, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_experts and hasattr(mlp, "experts"):
            self._check_fp8(mlp.experts)
            return self._sparse_block_fp8(mlp, x)
        experts = getattr(mlp, "experts", None)
        if (self.parallel_attention and self._fused and type(mlp).__name__ == "Qwen3_5MoeSparseMoeBlock"
                and experts is not None and x.is_cuda and torch.cuda.get_device_capability(x.device) == (9, 0)
                and x.dtype == torch.bfloat16 and x.shape[-1] == 2048 and x.is_contiguous()
                and _verified_grouped_bf16_build()
                and not torch.is_grad_enabled() and getattr(mlp.gate, "top_k", None) == 8
                and getattr(getattr(experts, "config", None), "_experts_implementation", None) == "grouped_mm"
                and experts.has_gate and not (experts.has_bias or experts.is_transposed or experts._is_expert_parallel)):
            from frida_decisions.v2.model.exec.bf16_experts import grouped_experts
            h = x.view(-1, x.shape[-1])
            shared = mlp.shared_expert(h)
            _, routing_weights, indices = mlp.gate(h)
            combined = grouped_experts(experts, h, indices, routing_weights)
            return (combined + torch.sigmoid(mlp.shared_expert_gate(h)) * shared).view_as(x)
        if self._fused and hasattr(mlp, "gate_proj") and type(mlp.down_proj) is torch.nn.Linear:
            gate, up = self._project((mlp.gate_proj, mlp.up_proj), x)
            if (self.parallel_attention and self._whole is not None and x.dtype == torch.bfloat16
                    and torch.cuda.get_device_capability(x.device) == (9, 0)
                    and getattr(self._stack.config, "hidden_act", None) == "silu" and gate.shape == up.shape):
                from frida_decisions.v2.model.exec.bf16_gate import gated
                active = gated(self._whole)
            else:
                active = mlp.act_fn(gate) * up
            return F.linear(active, mlp.down_proj.weight, mlp.down_proj.bias)
        out = mlp(x)
        return out[0] if isinstance(out, tuple) else out

    def _forward(self, plan: _Pass) -> torch.Tensor:
        stack = self._stack
        embeds = stack.get_input_embeddings()(plan.ids.unsqueeze(0))
        self._fused = self.fuses_norms and embeds.is_cuda and find_spec("triton") is not None
        if plan.images is not None:
            self._splice(embeds, plan.images)
        hidden = torch.cat([embeds, embeds.new_zeros(1, 1, embeds.shape[-1])], 1)
        positions = plan.positions.view(-1, 1, plan.positions.shape[-1]).expand(3, 1, -1)
        rotary = stack.rotary_emb(hidden, positions)
        readout = plan.readout
        for layer in stack.layers:
            mixed = self._recurrent(layer.linear_attn, self._norm(layer.input_layernorm, hidden), plan) \
                if layer.block_type == "linear_attention" else \
                self._attention(layer.self_attn, self._norm(layer.input_layernorm, hidden), plan, rotary,
                                skip_root=plan.skip_final_root and layer is stack.layers[-1])
            hidden = hidden + mixed
            if self.compact_readout and plan.skip_final_root and layer is stack.layers[-1]:
                # Only tokens consumed by the readout need the final token-local MLP.
                hidden = hidden.index_select(1, readout.col)
                compact = torch.arange(readout.col.numel(), device=hidden.device)
                readout = Readout(readout.row, compact, readout.slot, readout.n_slots)
            hidden = hidden + self._mlp(layer.mlp, self._norm(layer.post_attention_layernorm, hidden))
        return self.model.readout_head(self._norm(stack.norm, hidden), readout)

    def _splice(self, embeds: torch.Tensor, images: _Images) -> None:
        """The tower's features in place of the image tokens' embeddings."""
        features = self.model.vision.features({"pixel_values": torch.cat(images.pixel_values),
                                               "image_grid_thw": images.image_grid_thw,
                                               "host_grid": images.host_grid if self.parallel_attention else None})
        if features.shape[0] != images.at.numel():
            raise RuntimeError(f"{images.at.numel()} image tokens for {features.shape[0]} image features")
        embeds[0].index_copy_(0, images.at, features.to(embeds.device, embeds.dtype))

    def _recurrent(self, cell, x, plan: _Pass) -> torch.Tensor:
        kit = sys.modules[type(cell).__module__]
        n = x.shape[1]
        projected, raw_decay, raw_beta, gate = self._project(
            (cell.in_proj_qkv, cell.in_proj_a, cell.in_proj_b, cell.in_proj_z), x)
        whole = self._whole
        conv = kit.causal_conv1d_fn(projected[0].index_select(0, plan.conv_in).T.unsqueeze(0),
                                    cell.conv1d.weight.squeeze(1), cell.conv1d.bias,
                                    activation=cell.activation)
        parts = conv[0].T.split([cell.key_dim, cell.key_dim, cell.value_dim], -1)
        log_rate, shift = self._from_weights(cell, lambda: (cell.A_log.float(), cell.dt_bias.float()), x)
        floats = None
        repeat = cell.num_v_heads // cell.num_k_heads
        core = x.new_zeros(n, cell.num_v_heads, cell.head_v_dim)
        states = [None] * len(plan.nodes)
        for wave in plan.waves:
            shape, tokens = wave.tokens.shape, wave.tokens.view(-1)
            small = wave.cu is not None and wave.tokens.numel() <= _per_token_tokens() and wave.longest <= PER_TOKEN_ROW
            start = None if wave.parents is None else states[wave.level - 1].index_select(0, wave.parents)
            keep = wave.kept is not None
            if small and self._fused and whole is not None:
                at_decay = cell.in_proj_qkv.weight.shape[0]
                last = _norm_kernels()[2](conv[0], whole, log_rate, shift, tokens, wave.conv, wave.writes, core, start,
                                          keep, wave.cu, cell.num_k_heads, cell.num_v_heads, cell.head_k_dim,
                                          cell.head_v_dim, whole.shape[-1], at_decay,
                                          at_decay + cell.in_proj_a.weight.shape[0])
            else:
                if floats is None:
                    floats = raw_decay[0].float(), raw_beta[0]
                query = parts[0].index_select(0, wave.conv).reshape(*shape, -1, cell.head_k_dim)
                key = parts[1].index_select(0, wave.conv).reshape(*shape, -1, cell.head_k_dim)
                value = parts[2].index_select(0, wave.conv).reshape(*shape, -1, cell.head_v_dim)
                if repeat > 1:
                    query = query.repeat_interleave(repeat, dim=-2)
                    key = key.repeat_interleave(repeat, dim=-2)
                g = floats[0].index_select(0, tokens).view(*shape, -1)
                beta = floats[1].index_select(0, tokens).view(*shape, -1)
                if self._fused and wave.cu is not None:    # gate and sigmoid in the kernel
                    extra = dict(use_beta_sigmoid_in_kernel=True, use_gate_in_kernel=True, A_log=log_rate, dt_bias=shift)
                else:
                    extra = {}
                    g = -log_rate.exp() * F.softplus(g + shift)
                    if wave.cu is None:                    # pads read the empty token, whose decay must be 0
                        g = g * plan.decays[tokens].view(*shape, -1)
                    beta = beta.sigmoid()
                if wave.cu is None:
                    out, last = kit.torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=start,
                                                                 output_final_state=keep, use_qk_l2norm_in_kernel=True)
                    core.index_copy_(0, wave.at, out[wave.row, wave.col])
                else:
                    chunked, per_token = _fla_kernels()
                    out, last = (per_token if small else chunked)(
                        query, key, value, g=g, beta=beta, initial_state=start, output_final_state=keep,
                        use_qk_l2norm_in_kernel=True, cu_seqlens=wave.cu, **extra,
                        **({} if small else {"chunk_indices": wave.chunks}))
                    core.index_copy_(0, wave.at, out[0].index_select(0, wave.col))
            if keep:
                if states[wave.level] is None:
                    states[wave.level] = last.new_empty((plan.nodes[wave.level], *last.shape[1:]))
                states[wave.level].index_copy_(0, wave.kept[1], last.index_select(0, wave.kept[0]))
        gate = gate.reshape(-1, cell.head_v_dim)
        core = core.reshape(-1, cell.head_v_dim)
        if self._fused:
            core = _norm_kernels()[1](core, gate, cell.norm.weight, cell.norm.variance_epsilon)
        else:
            core = cell.norm(core, gate)
        return cell.out_proj(core.reshape(1, n, -1))

    def _attention(self, cell, x, plan: _Pass, rotary, skip_root=False) -> torch.Tensor:
        kit = sys.modules[type(cell).__module__]
        shape = (1, x.shape[1], -1, cell.head_dim)
        query, gate = torch.chunk(cell.q_proj(x).view(1, x.shape[1], -1, cell.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(1, x.shape[1], -1)
        query = self._norm(cell.q_norm, query.view(shape).contiguous()).transpose(1, 2)    # a half of `q_proj`
        key = self._norm(cell.k_norm, cell.k_proj(x).view(shape)).transpose(1, 2)
        value = cell.v_proj(x).view(shape).transpose(1, 2)
        query, key = kit.apply_rotary_pos_emb(query, key, *rotary)
        groups = cell.num_key_value_groups
        core = query.new_zeros(query.shape)
        def attend(a, scratch):
            # Final-layer root queries cannot affect descendant readouts. Descendants
            # still attend to the root keys and values projected from the layer input.
            if skip_root and a.keys is None:
                return scratch
            rows, width = a.queries.shape
            if ((self.compact_readout or self.parallel_attention) and a.contiguous_start is not None and width >= 2048
                    and x.is_cuda and (plan.images is None or self.parallel_attention)):
                # A single contiguous state needs neither indexed copies nor expanded KV heads.
                cols = slice(a.contiguous_start, a.contiguous_start + width)
                out = F.scaled_dot_product_attention(query[:, :, cols], key[:, :, cols], value[:, :, cols],
                                                     is_causal=True, scale=cell.scaling, enable_gqa=True)
                core[:, :, cols] = out
                return scratch
            q = query[0].index_select(1, a.queries.view(-1)).view(-1, rows, width, cell.head_dim).transpose(0, 1)
            take = a.queries if a.keys is None else a.keys
            flat, full = take.view(-1), take.shape[1]
            if a.kv_span is not None:
                prefix, capacity = a.kv_span
                if scratch is None:
                    # This layer and pass alone own these buffers. All operations use
                    # the current CUDA stream, so suffix writes follow the previous SDPA.
                    buffers = []
                    for source in (key, value):
                        buf = source.new_empty((source.shape[1], groups, capacity, cell.head_dim))
                        buf[:, :, :prefix].copy_(source[0, :, None, :prefix])
                        buffers.append(buf)
                    scratch = buffers
                if full > prefix:
                    suffix = flat[prefix:]
                    for buf, source in zip(scratch, (key, value)):
                        buf[:, :, prefix:full].copy_(source[0].index_select(1, suffix).unsqueeze(1))
                k, v = (buf.view(1, -1, capacity, cell.head_dim)[:, :, :full] for buf in scratch)
            else:
                k = kit.repeat_kv(key[0].index_select(1, flat).view(-1, rows, full, cell.head_dim).transpose(0, 1), groups)
                v = kit.repeat_kv(value[0].index_select(1, flat).view(-1, rows, full, cell.head_dim).transpose(0, 1), groups)
            mask = a.materialize_mask(x.device) if a.suffix_masks is not None else a.mask
            if mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=cell.scaling)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=cell.scaling)
            core[0].index_copy_(1, a.at, out[0].index_select(1, a.col) if rows == 1 else out[a.row, :, a.col].transpose(0, 1))
            return scratch

        scratch = None
        if (self.parallel_attention and x.is_cuda and len(plan.attends) >= 3
                and all(a.kv_span is not None for a in plan.attends)):
            # Each singleton writes disjoint rows of core. Its private scratch
            # belongs to one stream. Only CUDA stream handles survive the pass.
            main = torch.cuda.current_stream(x.device)
            count = min(4, len(plan.attends) - 1)
            stream_key = ("branch_streams", x.device, count)
            streams = self._cached.get(stream_key)
            if streams is None:
                streams = [torch.cuda.Stream(device=x.device) for _ in range(count)]
                self._cached[stream_key] = streams
            buffers = [None] * count
            for stream in streams:
                stream.wait_stream(main)
            try:
                attend(plan.attends[0], None)
                for index, row in enumerate(plan.attends[1:]):
                    slot = index % count
                    with torch.cuda.stream(streams[slot]):
                        buffers[slot] = attend(row, buffers[slot])
            finally:
                for stream in streams:
                    main.wait_stream(stream)
        else:
            for a in plan.attends:
                scratch = attend(a, scratch)
        core = core.transpose(1, 2).reshape(1, x.shape[1], -1)
        return cell.o_proj(core * torch.sigmoid(gate))


class CausalFlatPaddedExecutor(CausalFlatExecutor):
    """Rows of a wave padded to the longest on any device, for comparison."""
    concatenates = False
    fuses_norms = False


class CausalFlatFp8Executor(CausalFlatExecutor):
    """A MoE's experts in FP8 (`exec/fp8_experts.py`): changes the arithmetic, read its answers."""
    fp8_experts = True


class CausalFlatEagerExecutor(CausalFlatExecutor):
    """PyTorch's norms and their rounding, for comparison."""
    fuses_norms = False
