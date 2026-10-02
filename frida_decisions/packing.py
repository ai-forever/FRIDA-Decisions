"""Packing: every candidate of a request in one sequence, scored as if alone.

The naive way to score K options is K encoder passes over `[state][question]
[option_k]`, which re-encodes the state K times. Here a request becomes one
row

    [state] [question 1] [opt 1] [opt 2] ... [question 2] [opt 1] ...

with a three-level block mask:

* state tokens see only the state;
* a question's tokens see the state and the question itself;
* an option's tokens see the state, its own question and itself.

Options never see each other, so an option's score does not depend on which
other options arrived with it or in what order. Positions restart after each
question: every option of a question sits at the same positions, right after
the question, exactly where it would sit if it were scored alone. T5 has no
absolute position embeddings, only relative attention buckets, so the
positions enter through a `(L, L)` bucket matrix that is computed here and
handed to the encoder together with the mask.

A row is closed when it holds `max_options_per_row` options or the next
option would take it past `max_row_tokens`; the next row repeats the state and
the question. Both splits are exact, for the same reason as above.

This module is pure numpy so that the ONNX backend does not need torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .config import DecisionsConfig
from .protocol import Candidate


# ----------------------------------------------------------------- tokens
class TextEncoder:
    """Tokenizer wrapper: raw token ids, no special tokens, right truncation."""

    def __init__(self, tokenizer_file: str):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def encode(self, text: str, cap: int | None = None) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids if cap is None else ids[:cap]

    def count(self, text: str) -> int:
        return len(self.encode(text))


@dataclass
class TokenizedRequest:
    """A compiled request in token ids, with every segment already truncated."""
    state: list[int]
    questions: list[list[int]]                 # one id list per question, in order
    options: list[tuple[int, list[int]]]       # (question index, ids incl. EOS), in candidate order


def tokenize(candidates: list[Candidate], encoder: TextEncoder,
             config: DecisionsConfig, state_max: int | None = None) -> TokenizedRequest:
    state_max = config.state_max_tokens if state_max is None else state_max
    state = encoder.encode(candidates[0].state, state_max)
    questions, options, index = [], [], {}
    for cand in candidates:
        if cand.question_id not in index:
            index[cand.question_id] = len(questions)
            questions.append(encoder.encode(cand.instruction, config.instruction_max_tokens))
        body = encoder.encode(cand.text, config.option_max_tokens - 1) + [config.eos_token_id]
        options.append((index[cand.question_id], body))
    return TokenizedRequest(state, questions, options)


# ----------------------------------------------------------------- row layout
@dataclass
class Row:
    """Question and option tokens of one row; the state is not included.

    Block indices are local to the row. Positions are absolute, i.e. they
    already count the `prefix_len` state tokens in front of the row.
    """
    ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    questions: list[tuple[int, int]] = field(default_factory=list)            # (start, end)
    options: list[tuple[int, int, int, int]] = field(default_factory=list)    # (start, end, q_start, q_end)


def layout_rows(req: TokenizedRequest, budget_offset: int, max_options: int,
                max_tokens: int) -> list[Row]:
    """Split a request into rows. `budget_offset` is how many tokens a row holds
    before its first question: the state length when the state is packed into
    every row, zero when it comes from the state cache."""
    prefix_len = len(req.state)
    rows, row = [], Row()
    current_q, q_span, q_offset = None, None, None
    for qi, body in req.options:
        if qi != current_q:                     # a new question starts here
            current_q, q_span = qi, None
        q_body = req.questions[qi]
        pending = 0 if q_span is not None else len(q_body)
        n_opts = len(row.options)
        if n_opts >= max_options or (
                n_opts and budget_offset + len(row.ids) + pending + len(body) > max_tokens):
            rows.append(row)                    # full: continue in a new row
            row, q_span = Row(), None
        if q_span is None:                      # (re-)emit the question
            start = len(row.ids)
            row.ids += q_body
            row.positions += range(prefix_len, prefix_len + len(q_body))
            q_span = (start, len(row.ids))
            q_offset = prefix_len + len(q_body)
            row.questions.append(q_span)
        start = len(row.ids)
        row.ids += body
        row.positions += range(q_offset, q_offset + len(body))   # positions restart per option
        row.options.append((start, len(row.ids), *q_span))
    if row.options:
        rows.append(row)
    return rows


def count_packed_rows(req: TokenizedRequest, config: DecisionsConfig) -> int:
    """How many rows the packed path (state in every row) would build."""
    return len(layout_rows(req, len(req.state), config.max_options_per_row, config.max_row_tokens))


# ----------------------------------------------------------------- buckets
@lru_cache(maxsize=8)
def _distance_buckets(num_buckets: int, max_distance: int) -> np.ndarray:
    """Bucket of an absolute distance 0..max_distance (T5 bidirectional scheme,
    the negative half). Computed in float32, the precision the model was
    trained with; beyond `max_distance` every distance falls in the last bucket."""
    half = num_buckets // 2
    exact = half // 2
    n = np.arange(max_distance + 1, dtype=np.int64)
    ratio = np.log(np.maximum(n, 1).astype(np.float32) / np.float32(exact))
    scaled = ratio / np.log(np.float32(max_distance / exact)) * np.float32(half - exact)
    large = exact + scaled.astype(np.int64)
    return np.where(n < exact, n, np.minimum(large, half - 1)).astype(np.uint8)


def relative_buckets(q_pos: np.ndarray, k_pos: np.ndarray, num_buckets: int,
                     max_distance: int) -> np.ndarray:
    """`(len(q_pos), len(k_pos))` uint8 bucket matrix for arbitrary positions."""
    rel = k_pos[None, :].astype(np.int64) - q_pos[:, None].astype(np.int64)
    lut = _distance_buckets(num_buckets, max_distance)
    out = lut[np.minimum(np.abs(rel), max_distance)]
    return out + (rel > 0).astype(np.uint8) * np.uint8(num_buckets // 2)


# ----------------------------------------------------------------- tensors
@dataclass
class Readout:
    """Where each candidate's score is read: the mean over its own tokens."""
    row: np.ndarray       # (T,) row index of each read token
    col: np.ndarray       # (T,) column of each read token
    slot: np.ndarray      # (T,) which candidate the token belongs to
    count: int            # number of candidates


def _readout(spans: list[tuple[int, int, int]]) -> Readout:
    """`spans` = (row, start, end) per candidate, in candidate order."""
    rows, cols, slots = [], [], []
    for slot, (r, start, end) in enumerate(spans):
        rows += [r] * (end - start)
        cols += range(start, end)
        slots += [slot] * (end - start)
    return Readout(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                   np.asarray(slots, dtype=np.int64), len(spans))


def mean_readout(values: np.ndarray, readout: Readout) -> np.ndarray:
    """Average `values[row, col]` (any trailing shape) per candidate slot."""
    picked = values[readout.row, readout.col].astype(np.float64)
    sums = np.zeros((readout.count,) + picked.shape[1:], dtype=np.float64)
    np.add.at(sums, readout.slot, picked)
    counts = np.bincount(readout.slot, minlength=readout.count).astype(np.float64)
    return sums / counts.reshape((-1,) + (1,) * (picked.ndim - 1))


@dataclass
class PackedBatch:
    """Packed rows, each row = state + some of its questions and options."""
    input_ids: np.ndarray      # (R, L) int64
    buckets: np.ndarray        # (R, L, L) uint8
    allowed: np.ndarray        # (R, L, L) bool, the block mask
    readout: Readout
    lengths: list[int]         # unpadded length of every row


def build_rows(requests: list[TokenizedRequest], config: DecisionsConfig):
    """Lay out the packed rows of several requests.

    Returns `(rows, per_request)`: `rows` is a list of `(state ids, Row)` in
    candidate order, `per_request` the number of candidates of each request.
    Any contiguous slice of `rows` can be packed and run on its own; the
    margins of consecutive slices concatenate in candidate order.
    """
    rows, per_request = [], []
    for req in requests:
        layout = layout_rows(req, len(req.state), config.max_options_per_row,
                             config.max_row_tokens)
        rows += [(req.state, row) for row in layout]
        per_request.append(len(req.options))
    return rows, per_request


def pack(requests: list[TokenizedRequest], config: DecisionsConfig) -> PackedBatch:
    """All rows of several requests as one batch."""
    return pack_rows(build_rows(requests, config)[0], config)


def pack_rows(rows: list[tuple[list[int], Row]], config: DecisionsConfig) -> PackedBatch:
    """Materialise rows from `build_rows` as padded arrays."""
    width = max(len(state) + len(row.ids) for state, row in rows)
    width = -(-width // config.align) * config.align
    n = len(rows)
    input_ids = np.full((n, width), config.pad_token_id, dtype=np.int64)
    buckets = np.zeros((n, width, width), dtype=np.uint8)
    allowed = np.zeros((n, width, width), dtype=bool)
    spans, lengths = [], []
    for i, (state, row) in enumerate(rows):
        s = len(state)
        length = s + len(row.ids)
        lengths.append(length)
        input_ids[i, :s] = state
        input_ids[i, s:length] = row.ids
        pos = np.zeros(width, dtype=np.int64)
        pos[:s] = np.arange(s)
        pos[s:length] = row.positions
        buckets[i] = relative_buckets(pos, pos, config.num_buckets, config.max_distance)
        a = allowed[i]
        a[:s, :s] = True                                       # state sees the state
        for start, end in row.questions:                       # question: state + itself
            a[s + start:s + end, :s] = True
            a[s + start:s + end, s + start:s + end] = True
        for start, end, qs, qe in row.options:                 # option: state + question + itself
            a[s + start:s + end, :s] = True
            a[s + start:s + end, s + qs:s + qe] = True
            a[s + start:s + end, s + start:s + end] = True
            spans.append((i, s + start, s + end))
        pad = np.arange(length, width)
        a[pad, pad] = True             # a padding row must attend to something; nothing reads it
    return PackedBatch(input_ids, buckets, allowed, _readout(spans), lengths)


@dataclass
class QueryRows:
    """Question/option rows of one request, to run against a cached state."""
    input_ids: np.ndarray      # (R, W) int64
    buckets: np.ndarray        # (R, W, S + W) uint8, keys = cached state + row
    allowed: np.ndarray        # (R, W, S + W) bool
    readout: Readout
    rows: list[Row]


def pack_queries(req: TokenizedRequest, config: DecisionsConfig) -> QueryRows:
    """Rows without the state. The row budget no longer includes the state."""
    prefix = len(req.state)
    rows = layout_rows(req, 0, config.max_options_per_row, config.max_row_tokens)
    width = max(len(r.ids) for r in rows)
    width += (-(prefix + width)) % config.align         # keys = state + row, aligned
    n = len(rows)
    ids = np.full((n, width), config.pad_token_id, dtype=np.int64)
    buckets = np.zeros((n, width, prefix + width), dtype=np.uint8)
    allowed = np.zeros((n, width, prefix + width), dtype=bool)
    state_pos = np.arange(prefix, dtype=np.int64)
    spans = []
    for i, row in enumerate(rows):
        length = len(row.ids)
        ids[i, :length] = row.ids
        pos = np.zeros(width, dtype=np.int64)
        pos[:length] = row.positions
        buckets[i] = relative_buckets(pos, np.concatenate([state_pos, pos]),
                                      config.num_buckets, config.max_distance)
        a = allowed[i]
        for start, end in row.questions:
            a[start:end, :prefix] = True
            a[start:end, prefix + start:prefix + end] = True
        for start, end, qs, qe in row.options:
            a[start:end, :prefix] = True
            a[start:end, prefix + qs:prefix + qe] = True
            a[start:end, prefix + start:prefix + end] = True
            spans.append((i, start, end))
        pad = np.arange(length, width)
        a[pad, prefix + pad] = True                       # pads see themselves only
    return QueryRows(ids, buckets, allowed, _readout(spans), rows)


def state_buckets(n: int, config: DecisionsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Bucket matrix and mask for encoding a state of `n` tokens on its own,
    padded to the alignment (padding sees only itself)."""
    width = -(-n // config.align) * config.align
    pos = np.arange(width, dtype=np.int64)
    buckets = relative_buckets(pos, pos, config.num_buckets, config.max_distance)
    allowed = np.zeros((width, width), dtype=bool)
    allowed[:n, :n] = True
    pad = np.arange(n, width)
    allowed[pad, pad] = True
    return buckets, allowed
