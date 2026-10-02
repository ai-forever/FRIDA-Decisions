"""(a) The exported model against the original adapter implementation.

Skipped unless a reference implementation is available:

    FD_REFERENCE_JUDGE        a Python file defining `Judge(checkpoint, device=..., state_max=...)`
                              whose `.judge(request)` returns an object with `.margins`
                              (candidate order) and `.answers`
    FD_REFERENCE_CHECKPOINT   the adapter folder that was exported

Two variants of this package are compared with it, both in float32 on CPU:

* shipped: the exported folder as published (bfloat16 weights, upcast);
* exact:   the same adapter merged in float32 in memory, which isolates the
           code path from the bfloat16 rounding of the weights.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import sys

import pytest
from conftest import flat_margins, record

from frida_decisions.protocol import decision

REFERENCE = os.environ.get("FD_REFERENCE_JUDGE")
CHECKPOINT = os.environ.get("FD_REFERENCE_CHECKPOINT")

pytestmark = pytest.mark.skipif(not (REFERENCE and CHECKPOINT),
                                reason="FD_REFERENCE_JUDGE / FD_REFERENCE_CHECKPOINT not set")


@pytest.fixture(scope="module")
def reference(cases):
    """Reference margins and decisions for every case; the model is freed after."""
    spec = importlib.util.spec_from_file_location("reference_judge", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look their module up
    spec.loader.exec_module(module)
    judge = module.Judge(CHECKPOINT, device="cpu", state_max=384)
    out = []
    for case in cases:
        verdict = judge.judge(case["request"])
        out.append((list(verdict.margins),
                    {q: decision(a) for q, a in verdict.answers.items()}))
    del judge
    gc.collect()
    return out


def _compare(judge, cases, reference):
    drift, mismatches = 0.0, []
    per_case = {}
    for case, (ref_margins, ref_decisions) in zip(cases, reference):
        response = judge(case["request"])
        ours = flat_margins(response)
        assert len(ours) == len(ref_margins)
        d = max(abs(a - b) for a, b in zip(ours, ref_margins))
        per_case[case["name"]] = d
        drift = max(drift, d)
        for qid, answer in response["answers"].items():
            if decision(answer) != ref_decisions[qid]:
                mismatches.append(f"{case['name']}:{qid}")
    decisions = sum(len(d) for _, d in reference)
    return {"cases": len(cases), "decisions": decisions,
            "same_decisions": decisions - len(mismatches), "mismatches": mismatches,
            "max_margin_drift": drift,
            "worst_case": max(per_case, key=per_case.get)}


def test_exact_merge_matches_reference(model_dir, cases, reference):
    import torch

    from frida_decisions.modeling import DecisionEncoder
    from frida_decisions.torch_backend import Judge
    from tools.export_model import load_merged

    t5, head_state, _ = load_merged(CHECKPOINT)
    head = torch.nn.Linear(t5.config.d_model, 1)
    head.load_state_dict(head_state)
    judge = Judge(model_dir, DecisionEncoder(t5.eval(), head), "cpu", state_max=384)
    result = _compare(judge, cases, reference)
    record("parity_reference_exact_fp32", result)
    del judge, t5
    gc.collect()
    assert not result["mismatches"]
    assert result["max_margin_drift"] < 1e-3


def test_shipped_bf16_weights_match_reference(torch_judge, cases, reference):
    result = _compare(torch_judge, cases, reference)
    record("parity_reference_shipped_bf16", result)
    assert not result["mismatches"]
