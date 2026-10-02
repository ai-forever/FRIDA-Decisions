"""Evaluate the released artifacts on razvilka and measure them.

Runs the exported model through this package — torch on CUDA (bf16) and ONNX
int8 on CPU — on every razvilka item, scores both with `razvilka_eval.py`, and
records peak GPU memory and GPU latency for a 384-token text with 1 and 3
questions. With `--vllm URL`, also a running vLLM server (started with
`FRIDA_DECISIONS_STATE_MAX=512`, the state cut used here), 8 requests in flight.

    python tools/release_eval.py [model_dir] [razvilka.jsonl] [--skip-onnx] [--vllm URL]
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

VLLM = sys.argv[sys.argv.index("--vllm") + 1] if "--vllm" in sys.argv else None
args = [a for a in sys.argv[1:] if not a.startswith("--") and a != VLLM]
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


def vllm_predictions(url: str, items, state_max: int, workers: int = 8):
    """Answers from a vLLM server, checking that it cuts the state where `Judge` does."""
    import concurrent.futures as cf

    import httpx

    from frida_decisions.base import BaseJudge

    text = BaseJudge(Path(MODEL), state_max=state_max)
    client = httpx.Client(base_url=url, timeout=600)
    model = client.get("/v1/models").json()["data"][0]["id"]

    def one(item):
        request = item["request"] if isinstance(item["request"], dict) else json.loads(item["request"])
        (qid, _), = request["questions"].items()
        r = client.post("/pooling", json={"model": model, "data": request})
        r.raise_for_status()
        data = r.json()["data"]
        expected = len(text.compile(request)[2].state)
        if data["usage"]["state_tokens"] != expected:
            raise RuntimeError(f"{item['id']}: the server kept {data['usage']['state_tokens']} state "
                               f"tokens, Judge keeps {expected}: start it with "
                               f"FRIDA_DECISIONS_STATE_MAX={state_max}")
        return item["id"], data["answers"][qid], data["usage"]["cached_tokens"]

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, items))
    seconds = time.time() - t0
    return {k: a for k, a, _ in out}, seconds, sum(c for _, _, c in out)


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

    if VLLM:
        vpreds, seconds, cached = vllm_predictions(VLLM, items, 512)
        vscore = rz.score(vpreds, items)
        report["vllm"] = {"accuracy": vscore["accuracy"], "correct": vscore["correct"],
                          "per_type": vscore["per_type"],
                          "decisions_equal_to_torch": sum(
                              rz.decide(i, vpreds[i["id"]]) == rz.decide(i, gpu_decisions[i["id"]])
                              for i in items),
                          "different": [i["id"] for i in items if rz.decide(i, vpreds[i["id"]]) !=
                                        rz.decide(i, gpu_decisions[i["id"]])],
                          "requests_per_second_8_in_flight": round(len(items) / seconds, 1),
                          "cached_tokens": cached}

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
