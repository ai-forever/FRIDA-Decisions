"""(c) ONNX int8 (per-token activations) against torch float32, and CPU latency."""
from __future__ import annotations

import statistics
import time

import pytest
from conftest import THREADS, flat_margins, record

from frida_decisions.constants import ONNX_INT8_FILE, ONNX_SUBDIR
from frida_decisions.protocol import decision


@pytest.fixture(scope="module")
def onnx_judge(model_dir):
    pytest.importorskip("onnxruntime")
    if not (model_dir / ONNX_SUBDIR / ONNX_INT8_FILE).exists():
        pytest.skip("no ONNX graph in the model folder")
    from frida_decisions import OnnxJudge

    return OnnxJudge.from_pretrained(model_dir, threads=THREADS)


def typical_request(judge, i: int) -> dict:
    """A ~384-token state and three questions (yes/no, 3-way choice, 3-level score)."""
    base = ("Клиент пишет: списали 12 400 рублей дважды за один заказ, прошу вернуть "
            "деньги, жду ответа третий день, в чате поддержки никто не отвечает. ")
    state = base
    while judge.text.count(state) < 384:
        state += base
    return {"state": f"#{i} " + state, "questions": {
        "refund": {"type": "noul", "instructions": "Клиент просит вернуть деньги?"},
        "topic": {"type": "choice", "instructions": "Какая тема обращения?",
                  "criteria": {"billing": "оплата", "delivery": "доставка", "account": "аккаунт"}},
        "anger": {"type": "score", "instructions": "Насколько клиент раздражён?",
                  "criteria": ["спокоен", "раздражён", "в ярости"]}}}


def test_onnx_int8_agrees_with_torch(torch_judge, onnx_judge, cases):
    drift, total, same, mismatches = 0.0, 0, 0, []
    for case in cases:
        ref = torch_judge(case["request"])
        ours = onnx_judge(case["request"])
        drift = max(drift, max(abs(a - b) for a, b in zip(flat_margins(ref), flat_margins(ours))))
        for qid, answer in ref["answers"].items():
            total += 1
            if decision(answer) == decision(ours["answers"][qid]):
                same += 1
            else:
                mismatches.append(f"{case['name']}:{qid}")
    record("onnx_int8_vs_torch_fp32", {"cases": len(cases), "decisions": total, "same_decisions": same,
                                       "agreement": same / total, "mismatches": mismatches,
                                       "max_margin_drift": drift})
    assert same / total >= 0.9


def _median_ms(judge, make, runs: int = 5) -> float:
    judge(make(-1))                                   # warm-up
    times = []
    for i in range(runs):
        request = make(i)                             # a fresh state every run: no cache hits
        t = time.perf_counter()
        judge(request)
        times.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(times)


def test_cpu_latency(torch_judge, onnx_judge):
    make = lambda i: typical_request(torch_judge, i)  # noqa: E731
    _, _, tok = torch_judge.compile(make(0))
    result = {"threads": THREADS, "state_tokens": len(tok.state), "questions": 3,
              "candidates": len(tok.options),
              "torch_fp32_ms": round(_median_ms(torch_judge, make)),
              "onnx_int8_ms": round(_median_ms(onnx_judge, make))}
    result["speedup"] = round(result["torch_fp32_ms"] / result["onnx_int8_ms"], 2)
    record("cpu_latency_typical_request", result)
