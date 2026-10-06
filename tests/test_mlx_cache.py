"""Exact MLX state reuse, cache policy, and released model parity."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from conftest import record

mx = pytest.importorskip('mlx.core')
from frida_decisions import MlxJudge
from frida_decisions.mlx_modeling import StateCache
from frida_decisions.protocol import aggregate, decision
from test_mlx import tiny


def test_lru_storage_and_clear():
    cache = StateCache(64)
    k, v = [mx.zeros((4,), dtype=mx.float32)], [mx.zeros((4,), dtype=mx.float32)]
    for key in ((1,), (2,)):
        cache.put(key, k, v)
    assert cache.bytes == 64
    assert cache.get((1,)) is not None
    cache.put((3,), k, v)
    assert (2,) not in cache and (1,) in cache and (3,) in cache
    assert cache.evictions == 1
    cache.put((1,), k, v)
    assert cache.bytes == 64
    cache.put((4,), [mx.zeros((100,))], v)
    assert (4,) not in cache and cache.bytes == 64
    assert cache.get((2,)) is None
    assert (cache.hits, cache.misses) == (1, 1)
    cache.clear()
    assert cache.bytes == len(cache) == 0


@pytest.mark.parametrize('state', [[3, 4, 5], []])
@pytest.mark.parametrize('step', [1, 2, None])
def test_small_cached_parity_and_policy(tiny, model_dir, state, step):
    from frida_decisions.packing import TokenizedRequest
    judge = MlxJudge(model_dir, tiny, rows_per_forward=step)
    judge.config = replace(judge.config, max_options_per_row=2, max_row_tokens=32, align=8)
    req = TokenizedRequest(state, [[6, 7], [8]],
                           [(0, [9, 2]), (0, [10, 11, 2]), (1, [12, 2]), (1, [13, 2]), (1, [14, 2])])
    packed = judge._score_packed([req])[0][0]
    with patch.object(tiny, 'encode_state', wraps=tiny.encode_state) as encode:
        first, usage = judge._score([req])
        assert usage['state_cache'] == 'miss'
        second, usage = judge._score([req])
        assert usage['state_cache'] == 'hit' and encode.call_count == 1
        np.testing.assert_allclose(first[0], packed, atol=2e-6, rtol=0)
        np.testing.assert_allclose(second[0], packed, atol=2e-6, rtol=0)
        changed = replace(req, state=state + [15])
        assert judge._score([changed])[1]['state_cache'] == 'miss'
        assert encode.call_count == 2
        new_questions = replace(req, questions=[[17, 18]], options=[(0, [19, 2])])
        reused, usage = judge._score([new_questions])
        assert usage['state_cache'] == 'hit' and encode.call_count == 2
        np.testing.assert_allclose(reused[0], judge._score_packed([new_questions])[0][0],
                                   atol=2e-6, rtol=0)
    single = replace(req, state=[16], options=req.options[:1])
    assert judge._score([single])[1]['state_cache'] == 'off'
    assert judge._score([req, req])[1]['state_cache'] == 'off'
    judge.state_cache = None
    assert judge._score([req])[1]['state_cache'] == 'off'
    np.testing.assert_allclose(judge._score([req])[0][0], packed, atol=2e-6, rtol=0)


def test_disabled_and_invalid_budget(tiny, model_dir):
    assert MlxJudge(model_dir, tiny, state_cache_mb=0).state_cache is None
    with pytest.raises(ValueError):
        MlxJudge(model_dir, tiny, state_cache_mb=-1)


def cache_parity(model_dir, cases):
    judge = MlxJudge.from_pretrained(model_dir, state_max=512)
    worst, total, per_case = 0.0, 0, {}
    for case in cases:
        req = case['request']
        parsed, candidates, tok = judge.compile(req)
        packed = judge.margins_packed(req)
        judge.state_cache.clear()
        with patch.object(judge.model, 'encode_state', wraps=judge.model.encode_state) as encode:
            cached, usage = judge._score_cached(tok)
            assert usage['state_cache'] == 'miss'
            again, usage = judge._score_cached(tok)
            assert usage['state_cache'] == 'hit' and encode.call_count == 1
        assert len(cached) == len(packed) == len(candidates)
        assert np.isfinite(cached).all()
        np.testing.assert_allclose(cached, packed, atol=1e-3, rtol=0)
        np.testing.assert_allclose(again, packed, atol=1e-3, rtol=0)
        a, b = aggregate(parsed, candidates, packed), aggregate(parsed, candidates, cached)
        assert {q: decision(v) for q, v in a.items()} == {q: decision(v) for q, v in b.items()}
        for q in a:
            if a[q]['type'] == 'ranking':
                assert a[q]['ranking'] == b[q]['ranking']
        drift = float(np.max(np.abs(np.array(cached) - packed)))
        worst = max(worst, drift)
        total += len(a)
        per_case[case['name']] = drift
    return {'requests': len(cases), 'decisions': total,
        'max_margin_drift': worst, 'decision_mismatches': 0, 'per_case': per_case,
        'hits': judge.state_cache.hits, 'misses': judge.state_cache.misses}


def test_released_cache_parity(model_dir, cases):
    record('mlx_cache_parity', cache_parity(model_dir, cases))



def test_public_catalog_cache_parity(model_dir):
    path = Path(__file__).resolve().parents[1] / 'examples/data/intent_catalog.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    record('mlx_public_catalog_cache_parity', cache_parity(model_dir, [
        {'name': 'public/intent-catalog', 'request': catalog['request']}]))
