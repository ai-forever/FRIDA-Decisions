"""(b) The state cache and packing are exact: same margins as scoring alone.

* cache vs packed: the state encoded once and reused, against the state
  packed into every row;
* packed vs naive: every candidate scored in a sequence of its own
  (`[state][question][option]`), against the packed rows. Run on the
  repository's own cases only (the naive path is slow on CPU).
"""
from __future__ import annotations

import time

from conftest import load_cases, record

from frida_decisions.packing import TokenizedRequest
from frida_decisions.protocol import aggregate, decision


def _decisions(judge, request, margins):
    parsed, candidates, _ = judge.compile(request)
    return {q: decision(a) for q, a in aggregate(parsed, candidates, margins).items()}


def test_cache_matches_packed(torch_judge, cases):
    drift, mismatches, timings = 0.0, [], {}
    for case in cases:
        torch_judge.state_cache.clear()
        t0 = time.perf_counter()
        packed = torch_judge.margins_packed(case["request"])
        t1 = time.perf_counter()
        cached = torch_judge.margins_cached(case["request"])          # miss: state encoded
        t2 = time.perf_counter()
        torch_judge.margins_cached(case["request"])                   # hit: state reused
        t3 = time.perf_counter()
        drift = max(drift, max(abs(a - b) for a, b in zip(packed, cached)))
        if _decisions(torch_judge, case["request"], packed) != _decisions(torch_judge, case["request"], cached):
            mismatches.append(case["name"])
        timings[case["name"]] = {"packed_ms": round((t1 - t0) * 1e3), "cache_miss_ms": round((t2 - t1) * 1e3),
                                 "cache_hit_ms": round((t3 - t2) * 1e3)}
    torch_judge.state_cache.clear()
    record("cache_vs_packed_fp32_cpu", {"cases": len(cases), "max_margin_drift": drift,
                                        "decision_mismatches": mismatches, "timings": timings})
    assert not mismatches
    assert drift < 1e-3


def test_packed_matches_naive(torch_judge):
    cases = load_cases()[:9]                     # the repository's own cases
    drift = 0.0
    for case in cases:
        _, _, tok = torch_judge.compile(case["request"])
        packed = torch_judge._score_packed([tok])[0][0]
        singles = [TokenizedRequest(tok.state, tok.questions, [option]) for option in tok.options]
        naive = [m for single in singles for m in torch_judge._score_packed([single])[0][0]]
        drift = max(drift, max(abs(a - b) for a, b in zip(packed, naive)))
    record("packed_vs_naive_fp32_cpu", {"cases": len(cases), "max_margin_drift": drift})
    assert drift < 1e-3


def test_batch_matches_single(torch_judge):
    cases = load_cases()[:9]
    requests = [c["request"] for c in cases]
    batched = torch_judge.judge_batch(requests)
    torch_judge.state_cache.clear()
    drift = 0.0
    for request, response in zip(requests, batched):
        alone = torch_judge.margins_packed(request)
        ours = [m for q in response["margins"].values() for m in q.values()]
        drift = max(drift, max(abs(a - b) for a, b in zip(alone, ours)))
    record("batch_vs_single_fp32_cpu", {"requests": len(requests), "max_margin_drift": drift})
    assert drift < 1e-3
