"""Packing layout: buckets, block mask, row splitting (no model needed)."""
from __future__ import annotations

import numpy as np
import pytest

from frida_decisions.config import DecisionsConfig
from frida_decisions.packing import (TokenizedRequest, layout_rows, pack, pack_queries,
                                     relative_buckets)


def reference_buckets(rel, num_buckets=32, max_distance=128):
    """T5 bidirectional bucketing written out in float32 torch, the training-time formula."""
    torch = pytest.importorskip("torch")
    rel = torch.as_tensor(rel)
    half = num_buckets // 2
    out = (rel > 0).long() * half
    n = rel.abs()
    exact = half // 2
    large = exact + (torch.log(n.clamp(min=1).float() / exact)
                     / torch.log(torch.tensor(max_distance / exact)) * (half - exact)).long()
    return (out + torch.where(n < exact, n, large.clamp(max=half - 1))).numpy()


def test_buckets_match_the_float32_formula():
    pos = np.arange(3000)
    ours = relative_buckets(pos[:300], pos, 32, 128)
    rel = pos[None, :] - pos[:300, None]
    assert (ours == reference_buckets(rel)).all()


def _request():
    # a 5-token state, two questions (3 and 2 tokens), options of 2-4 tokens incl. EOS
    return TokenizedRequest(state=[10, 11, 12, 13, 14], questions=[[20, 21, 22], [30, 31]],
                            options=[(0, [40, 2]), (0, [41, 42, 2]), (0, [43, 2]),
                                     (1, [50, 51, 52, 2]), (1, [53, 2])])


def test_mask_and_positions():
    batch = pack([_request()], DecisionsConfig(align=8))
    assert batch.input_ids.shape == (1, 24)                      # 5 + 3 + 7 + 2 + 6 = 23 -> 24
    a = batch.allowed[0]
    s = 5
    assert a[:s, :s].all() and not a[:s, s:].any()               # state sees only the state
    q1 = slice(s, s + 3)
    assert a[q1, :s].all() and a[q1, q1].all() and not a[q1, s + 3:].any()
    o1, o2 = slice(8, 10), slice(10, 13)
    assert a[o1, :s].all() and a[o1, q1].all() and a[o1, o1].all()
    assert not a[o1, o2].any() and not a[o2, o1].any()           # options never see each other
    # positions restart per option: both start at the same distance from the state
    assert (batch.buckets[0, 8, :s] == batch.buckets[0, 10, :s]).all()
    assert batch.readout.count == 5


def test_row_split_by_count_repeats_the_question():
    req = TokenizedRequest([1, 2], [[3]], [(0, [5, 2])] * 5)
    rows = layout_rows(req, budget_offset=2, max_options=2, max_tokens=1024)
    assert [len(r.options) for r in rows] == [2, 2, 1]
    assert all(r.ids[0] == 3 for r in rows)


def test_row_split_by_tokens_keeps_at_least_one_option():
    req = TokenizedRequest([1] * 10, [[3]], [(0, [5] * 50 + [2])] * 3)
    rows = layout_rows(req, budget_offset=10, max_options=16, max_tokens=40)
    assert [len(r.options) for r in rows] == [1, 1, 1]


def test_query_rows_align_keys():
    q = pack_queries(_request(), DecisionsConfig(align=8))
    assert q.allowed.shape[2] % 8 == 0
    assert q.allowed[0, :, :5].any()
