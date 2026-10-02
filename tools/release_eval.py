"""Evaluate the released artifacts on razvilka and measure them.

Runs the exported model through this package — torch on CUDA (bf16) and ONNX
int8 on CPU — on every razvilka item, scores both with `razvilka_eval.py`, and
records peak GPU memory and GPU latency for a 384-token text with 1 and 3
questions.

    python tools/release_eval.py [model_dir] [razvilka.jsonl] [--skip-onnx]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks" / "razvilka")]

import torch  # noqa: E402

import razvilka_eval as rz  # noqa: E402
from frida_decisions import Judge, OnnxJudge  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = args[0] if args else str(ROOT / "_export" / "FRIDA-Decisions")
DATA = args[1] if len(args) > 1 else None
OUT = ROOT / "tests" / "_results" / "release_eval.json"


def predictions(judge, items):
    preds = {}
    for item in items:
        request = item["request"] if isinstance(item["request"], dict) else json.loads(item["request"])
        (qid, _), = request["questions"].items()
        preds[item["id"]] = judge.judge(request)["answers"][qid]
    return preds


def typical(questions: int, i: int) -> dict:
    state = f"Обращение {i}. " + "Пишу по поводу домашнего интернета, связь рвётся по вечерам, прошу перерасчёт. " * 22
    qs = {"topic": {"type": "choice", "instructions": "Тема обращения?",
                    "criteria": {"internet": "домашний интернет", "mobile": "мобильная связь", "tv": "телевидение"}},
          "refund": {"type": "noul", "instructions": "Клиент просит перерасчёт?"},
          "mood": {"type": "score", "instructions": "Насколько клиент раздражён?",
                   "criteria": ["спокоен", "недоволен", "в ярости"]}}
    return {"state": state, "questions": dict(list(qs.items())[:questions])}


def gpu_latency(judge, questions: int) -> float:
    for i in range(3):
        judge.judge(typical(questions, -1 - i))
    times = []
    for i in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        judge.judge(typical(questions, i)); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1)


def main() -> None:
    items = rz.load(DATA) if DATA else rz.load()
    report = {"items": len(items)}

    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(free - 1.5 * 2**30, 2**30) / total)
    torch.cuda.reset_peak_memory_stats()
    gpu = Judge.from_pretrained(MODEL, device="cuda", state_max=512)
    preds = predictions(gpu, items)
    score = rz.score(preds, items)
    report["torch_cuda_bf16"] = {"accuracy": score["accuracy"], "correct": score["correct"],
                                 "per_type": score["per_type"],
                                 "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
    report["gpu_latency_ms_384_tokens"] = {"1_question": gpu_latency(gpu, 1),
                                           "3_questions": gpu_latency(gpu, 3),
                                           "tokens_3q": gpu.token_counts(typical(3, 0))}
    report["gpu"] = torch.cuda.get_device_name(0)
    gpu_decisions = {k: v for k, v in preds.items()}
    del gpu
    torch.cuda.empty_cache()

    if "--skip-onnx" not in sys.argv:
        onnx = OnnxJudge.from_pretrained(MODEL, state_max=512, threads=6)
        t0 = time.time()
        opreds = predictions(onnx, items)
        oscore = rz.score(opreds, items)
        agree = sum(rz.decide(i, opreds[i["id"]]) == rz.decide(i, gpu_decisions[i["id"]])
                    for i in items) if hasattr(rz, "decide") else None
        report["onnx_int8_cpu"] = {"accuracy": oscore["accuracy"], "correct": oscore["correct"],
                                   "per_type": oscore["per_type"],
                                   "decisions_equal_to_torch": agree,
                                   "minutes": round((time.time() - t0) / 60, 1)}

    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
