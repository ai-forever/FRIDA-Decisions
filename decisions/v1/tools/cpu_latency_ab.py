"""Interleaved CPU latency A/B: torch fp32 vs ONNX int8 on the same requests.

Alternating the two backends request by request keeps background load from
favouring either side; the ratio is the number to trust on a busy machine.

    python tools/cpu_latency_ab.py [model_dir] [threads]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from frida_decisions import Judge, OnnxJudge  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "_export" / "FRIDA-Decisions")
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 6
torch.set_num_threads(threads)


def request(i: int) -> dict:
    state = f"Обращение {i}. " + "Пишу по поводу домашнего интернета, связь рвётся по вечерам, прошу перерасчёт. " * 22
    return {"state": state, "questions": {
        "topic": {"type": "choice", "instructions": "Тема обращения?",
                  "criteria": {"internet": "домашний интернет", "mobile": "мобильная связь", "tv": "телевидение"}},
        "refund": {"type": "noul", "instructions": "Клиент просит перерасчёт?"},
        "mood": {"type": "score", "instructions": "Насколько клиент раздражён?",
                 "criteria": ["спокоен", "недоволен", "в ярости"]}}}


tj = Judge.from_pretrained(model, device="cpu", dtype=torch.float32)
oj = OnnxJudge.from_pretrained(model, threads=threads)
for j in (tj, oj):
    j.judge(request(0))
t_ms, o_ms = [], []
for i in range(1, 8):
    for j, sink in ((tj, t_ms), (oj, o_ms)):
        t0 = time.perf_counter(); j.judge(request(i)); sink.append((time.perf_counter() - t0) * 1000)
res = {"threads": threads, "state_tokens": tj.token_counts(request(1))["state_tokens"],
       "torch_fp32_ms_median": round(statistics.median(t_ms)), "onnx_int8_ms_median": round(statistics.median(o_ms)),
       "speedup_median": round(statistics.median(t_ms) / statistics.median(o_ms), 2),
       "torch_all": [round(x) for x in t_ms], "onnx_all": [round(x) for x in o_ms]}
print(json.dumps(res, ensure_ascii=False, indent=1))
(ROOT / "tests" / "_results").mkdir(exist_ok=True)
(ROOT / "tests" / "_results" / "cpu_latency_ab.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
