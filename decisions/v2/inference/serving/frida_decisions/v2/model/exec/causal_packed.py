"""The tree visibility over packed sequences, with a FlexAttention `BlockMask`.

`CausalPackedExecutor` takes the rows `CausalTreeExecutor` would build
(`state | question | options...`, a request split over several rows, the state
repeated in each, when it passes `pack_tokens`) and packs the rows of several
requests into sequences of a fixed length: no padding inside a sequence, only
at its tail. Positions restart per row exactly as in the tree (RoPE reads
`position_ids`), and the mask is a `BlockMask` built on the device from three
per-token tensors:

* `node[t]` -- the token's segment within its row (a node id unique in the
  batch: the same segment repeated in two rows is two nodes);
* `depth[t]` -- that segment's depth (state 0, question 1, option 2, ...);
* `anc[t, d]` -- the node of the token's ancestor-or-self at depth `d`, or -1.

`kv` is visible to `q` iff `anc[q, depth[kv]] == node[kv]` (kv's segment is an
ancestor of q's, or q's own) and, within q's own segment, `kv <= q`. That is
the tree's `allowed(i, j)`; a different row's nodes never match, so rows in
one sequence do not see each other. Every tail-padding token is a node of
its own, so it sees only itself (softmax stays defined) and the tail's
blocks are empty off the diagonal: flex skips them.

A sequence is as long as the smallest bucket that holds its rows: a power
of two from 1024 up to `pack_tokens` (in multiples of 128), so the tail
stays short and flex compiles a handful of shapes.

The attention. HF Qwen3 hands a 4D mask -- a `BlockMask` too -- to its
attention function untouched (`masking_utils` returns 4D masks as they are),
and looks that function up by `config._attn_implementation` at every call,
including the checkpoint recompute in backward. So the backbone is switched,
once and for good, to a registered implementation that sends a `BlockMask` to
`flex_attention` and every other mask to the backbone's original one (and
whose mask builder is the original's): the other executors see no change.
FlexAttention runs compiled on CUDA (`torch.compile(flex_attention)`, as HF
does). Off CUDA, where flex has no backward, the same `mask_mod` is expanded
to a dense boolean mask for SDPA: the CPU tests check the layout and the
mask, not the kernel.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch

from frida_decisions.v2.model.exec.causal import CONFIG, CausalBatch, _bases
from frida_decisions.v2.model.head import Readout
from frida_decisions.v2.model.layout import ScoringPlan, aligned, layout_rows

BLOCK = 128                  # FlexAttention's block size: sequences are a multiple of it
MIN_BUCKET = 1024            # the shortest sequence length
# Triton tiles that fit sm90's 227 KiB of shared memory at head_dim 128 in
# training: torch 2.9's defaults ask ~290 KiB and fail to compile on an H100
# ("No valid triton configs ... out of resource"). Forward: BLOCK_M x BLOCK_N;
# backward: BLOCK_M1 x BLOCK_N1 (dK, dV), BLOCK_M2 x BLOCK_N2 (dQ). All divide
# the mask's BLOCK. A config's `flex_kernel_options` replaces this.
FLEX_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64,
                       "BLOCK_M2": 64, "BLOCK_N2": 32, "num_stages": 1}


@dataclass
class _Row:
    ids: list[int]
    pos: list[int]
    node: list[int]          # row-local node per token: 0 the state, k the k-th block
    depth: list[int]         # per row-local node
    anc: list[list[int]]     # per row-local node: its ancestors-or-self, root first
    reads: list[tuple[int, int, int]]    # (start, end, global slot)


@dataclass
class PackedMicroBatch:
    input_ids: torch.Tensor                    # (b, L)
    position_ids: torch.Tensor                 # (b, L)
    node: torch.Tensor                         # (b, L) node id, unique in the batch
    depth: torch.Tensor                        # (b, L)
    anc: torch.Tensor                          # (b, L, D) ancestor-or-self node per depth, -1 none
    readout: Readout                           # rows = sequences
    slots: torch.Tensor                        # (m,) global slot of each local slot

    def to(self, device):
        return PackedMicroBatch(self.input_ids.to(device), self.position_ids.to(device),
                                self.node.to(device), self.depth.to(device), self.anc.to(device),
                                self.readout.to(device), self.slots.to(device))


class CausalPackedExecutor:
    """Tree rows of several requests packed into fixed-length sequences, with a
    FlexAttention `BlockMask` (see the module doc).

    `model.config.pack_tokens` is the longest sequence (rounded up to a
    multiple of 128; a row longer than that gets a longer sequence of its own)
    and the row cap: a request whose state and options pass it is split over
    rows like the tree's `max_row_tokens`. A sequence is padded to the
    smallest bucket that holds it (`bucket`). `max_batch_tokens` caps
    sequences x length per forward (None: one forward); a forward holds
    sequences of one length. `cost` is the unpadded token count."""
    # No KV cache: transformers' per-layer checkpointing applies as is (the
    # BlockMask is a forward argument; the recompute gets the same one).
    trains_with_checkpointing = True

    def __init__(self, model, max_batch_tokens=CONFIG):
        self.model = model
        self._budget = max_batch_tokens

    @property
    def max_batch_tokens(self) -> int | None:
        if self._budget is CONFIG:
            return self.model.config.max_batch_tokens
        return self._budget

    @property
    def pack_tokens(self) -> int:
        return aligned(self.model.config.pack_tokens, BLOCK)

    @property
    def kernel_options(self) -> dict:
        options = getattr(self.model.config, "flex_kernel_options", None)
        return dict(FLEX_KERNEL_OPTIONS if options is None else options)

    # ------------------------------------------------------------ layout
    def rows(self, plans: Sequence[ScoringPlan]) -> list[_Row]:
        slot_base, _ = _bases(plans)
        out = []
        for p, plan in enumerate(plans):
            root = plan.root
            s = len(root.ids)
            for row in layout_rows(plan, s, 10**9, self.pack_tokens):
                local = {seg: k + 1 for k, (_, _, seg) in enumerate(row.blocks)}
                node, depth, anc = [0] * s, [0], [[0]]
                for start, end, seg in row.blocks:
                    chain = plan.chain(seg)            # below the root, ending with seg
                    node += [local[seg]] * (end - start)
                    depth.append(len(chain))
                    anc.append([0] + [local[a] for a in chain])
                out.append(_Row(list(root.ids) + row.ids, list(root.pos) + row.positions,
                                node, depth, anc,
                                [(s + a, s + b, slot_base[p] + slot) for a, b, slot in row.reads]))
        return out

    def cost(self, plans: Sequence[ScoringPlan]) -> int:
        """Unpadded tokens of the forward: what a rank's share is balanced by."""
        return sum(len(r.ids) for r in self.rows(plans))

    def bucket(self, tokens: int) -> int:
        """The padded length of a sequence holding `tokens`: the smallest power
        of two from `MIN_BUCKET` that holds it, capped at `pack_tokens` (a
        longer single row: the next power of two)."""
        cap = self.pack_tokens
        length = min(MIN_BUCKET, cap)
        while length < tokens:
            length *= 2
        return length if tokens > cap else min(length, cap)

    def _bins(self, rows: list[_Row]) -> list[list[int]]:
        """First-fit decreasing into sequences of `pack_tokens`; a longer row alone."""
        cap = self.pack_tokens
        bins, room = [], []
        for i in sorted(range(len(rows)), key=lambda i: -len(rows[i].ids)):
            n = len(rows[i].ids)
            for b, free in enumerate(room):
                if n <= free:
                    bins[b].append(i)
                    room[b] -= n
                    break
            else:
                bins.append([i])
                room.append(cap - n)
        return bins

    def pack(self, plans: Sequence[ScoringPlan]) -> CausalBatch:
        from frida_decisions.v2.model.exec.causal import text_only
        text_only(plans, self)
        rows = self.rows(plans)
        bins = self._bins(rows)
        lengths = [self.bucket(sum(len(rows[i].ids) for i in b)) for b in bins]
        budget = self.max_batch_tokens
        groups, current = [], []                    # sequences per forward, same length each
        for b in sorted(range(len(bins)), key=lambda b: lengths[b]):
            if current and (lengths[b] != lengths[current[0]] or
                            (budget is not None and (len(current) + 1) * lengths[b] > budget)):
                groups.append(current)
                current = []
            current.append(b)
        if current:
            groups.append(current)
        depth_max = max(max(r.depth) for r in rows) + 1
        pad = self.model.pad_token_id
        micro, next_node = [], 0
        for group in groups:
            b, width = len(group), lengths[group[0]]
            ids = torch.full((b, width), pad, dtype=torch.long)
            pos = torch.zeros((b, width), dtype=torch.long)
            node = torch.empty((b, width), dtype=torch.long)
            depth = torch.zeros((b, width), dtype=torch.long)
            anc = torch.full((b, width, depth_max), -1, dtype=torch.long)
            spans, slots = [], []
            for r, k in enumerate(group):
                at = 0
                for i in bins[k]:
                    row = rows[i]
                    n = len(row.ids)
                    ids[r, at:at + n] = torch.tensor(row.ids)
                    pos[r, at:at + n] = torch.tensor(row.pos)
                    local = torch.tensor(row.node)
                    node[r, at:at + n] = next_node + local
                    depth[r, at:at + n] = torch.tensor(row.depth)[local]
                    table = torch.full((len(row.depth), depth_max), -1, dtype=torch.long)
                    for j, chain in enumerate(row.anc):
                        table[j, :len(chain)] = next_node + torch.tensor(chain)
                    anc[r, at:at + n] = table[local]
                    next_node += len(row.depth)
                    for start, end, slot in row.reads:
                        spans.append((r, at + start, at + end))
                        slots.append(slot)
                    at += n
                tail = next_node + torch.arange(width - at)    # the tail: a node per token
                node[r, at:] = tail
                anc[r, at:, 0] = tail
                next_node += width - at
            order = sorted(range(len(slots)), key=lambda j: slots[j])
            micro.append(PackedMicroBatch(
                ids, pos, node, depth, anc, Readout.from_spans([spans[j] for j in order]),
                torch.tensor([slots[j] for j in order], dtype=torch.long)))
        _, group = _bases(plans)
        return CausalBatch(micro, group, [len(r.ids) for r in rows],
                           sum(p.n_slots for p in plans))

    # ----------------------------------------------------------- forward
    def forward(self, batch: CausalBatch) -> torch.Tensor:
        """One margin per slot, in slot order (the `Compiled.flat()` order)."""
        backbone, head = self.model.backbone, self.model.readout_head
        _route_block_masks(backbone)
        options = self.kernel_options
        scores, slots = [], []
        for mb in batch.micro:
            mask = block_mask(mb.node, mb.depth, mb.anc)
            # HF hands extra keyword arguments down to the attention function,
            # in the checkpoint recompute too.
            hidden = backbone(input_ids=mb.input_ids, position_ids=mb.position_ids,
                              attention_mask=mask, use_cache=False,
                              kernel_options=options).last_hidden_state
            scores.append(head(hidden, mb.readout))
            slots.append(mb.slots)
        scores, slots = torch.cat(scores), torch.cat(slots)
        return scores[torch.argsort(slots)]

    def score(self, plans: Sequence[ScoringPlan]) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        margins = self.forward(self.pack(plans).to(device))
        return list(margins.split([p.n_slots for p in plans]))


def block_mask(node: torch.Tensor, depth: torch.Tensor, anc: torch.Tensor):
    """The `BlockMask` of the module doc over `(b, L)` sequences."""
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q, kv):
        kv_node = node[b, kv]
        return (anc[b, q, depth[b, kv]] == kv_node) & ((kv_node != node[b, q]) | (kv <= q))

    b, width = node.shape
    return create_block_mask(mask_mod, b, None, width, width, device=node.device,
                             BLOCK_SIZE=BLOCK, _compile=node.device.type == "cuda")


ROUTE = "fdd_blockmask_"


def _route_block_masks(backbone) -> None:
    """Switch the backbone to `ROUTE + <its implementation>` (see the module doc)."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    config = backbone.config
    original = config._attn_implementation
    if original.startswith(ROUTE):
        return
    name = ROUTE + original
    if name not in ALL_ATTENTION_FUNCTIONS:
        AttentionInterface.register(name, _routed_attention(original))
        AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS[original])
    config._attn_implementation_internal = name


@lru_cache(maxsize=None)
def _routed_attention(original: str):
    from torch.nn.attention.flex_attention import BlockMask, create_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if isinstance(attention_mask, BlockMask):
            if dropout:
                raise ValueError("the packed executor runs flex attention without dropout")
            gqa = key.shape[1] != query.shape[1]
            if query.device.type == "cuda":
                out = _flex()(query, key, value, block_mask=attention_mask, scale=scaling,
                              enable_gqa=gqa, kernel_options=kwargs.get("kernel_options"))
            else:                                  # flex has no backward off CUDA
                b, _, q_len, kv_len = attention_mask.shape
                dense = create_mask(attention_mask.mask_mod, b, 1, q_len, kv_len, query.device)
                out = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=dense, scale=scaling, enable_gqa=gqa)
            return out.transpose(1, 2).contiguous(), None
        if original == "eager":
            fn = sys.modules[type(module).__module__].eager_attention_forward
        else:
            fn = ALL_ATTENTION_FUNCTIONS[original]
        return fn(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                  **kwargs)

    return attention


@lru_cache(maxsize=None)
def _flex():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention, dynamic=False)
