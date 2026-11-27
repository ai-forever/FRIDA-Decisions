"""GPU check of the async API: `VllmJudge` (in-process vLLM) or `AsyncJudge` (torch).

    python tools/vllm_parity.py --reference-only              # float32 reference on CPU, once
    python tools/async_check.py vllm  [--gpu-memory-utilization 0.45]
    python tools/async_check.py torch

Both: every test case awaited at once, against `Judge` in float32 (decisions and
the largest margin difference); then 256 requests awaited at once (distinct
texts) for throughput. `vllm` also checks that a repeated text comes from the
cache, that another `cache_salt` does not, and that cancelling a request leaves
the engine working.

Writes `tests/_results/async_check_<mode>.json`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests")]

RESULTS = ROOT / "tests" / "_results"


def flat(margins: dict) -> list[float]:
    return [m for per_q in margins.values() for m in per_q.values()]


def compare(cases, out, ref) -> dict:
    from frida_decisions.protocol import decision

    same = total = 0
    drift = 0.0
    for case, response in zip(cases, out):
        theirs = ref[case["name"]]
        drift = max(drift, max(abs(a - b) for a, b in zip(flat(response["margins"]), flat(theirs["margins"]))))
        answers = response["answers"]
        for q, answer in theirs["answers"].items():
            total += 1
            same += decision(answer) == decision(answers[q])
    return {"same_decisions": f"{same}/{total}", "max_margin_drift": drift}


async def burst(judge, case, n: int) -> dict:
    requests = [dict(case, state=f"Обращение №{i}. " + case["state"]) for i in range(n)]
    started = time.perf_counter()
    out = await asyncio.gather(*(judge.judge(r) for r in requests))
    seconds = time.perf_counter() - started
    return {"requests": n, "seconds": round(seconds, 2), "requests_per_second": round(n / seconds, 1),
            "all_answered": all("answers" in o for o in out)}


async def check(mode: str, args) -> dict:
    import conftest
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)    # conftest hides CUDA for the CPU tests

    cases, model_dir = conftest.load_cases(), conftest.MODEL_DIR
    ref = json.loads((RESULTS / "vllm_reference_fp32.json").read_text(encoding="utf-8"))
    ref = {name: {"margins": e["margins"], "answers": e.get("answers")} for name, e in ref.items()}
    if any(v["answers"] is None for v in ref.values()):
        raise SystemExit("rerun tools/vllm_parity.py --reference-only: the reference has no answers")
    report = {"mode": mode, "cases": len(cases)}
    if mode == "vllm":
        from frida_decisions import VllmJudge
        judge = VllmJudge.from_pretrained(str(model_dir), gpu_memory_utilization=args.gpu_memory_utilization)
    else:
        from frida_decisions import AsyncJudge
        judge = AsyncJudge.from_pretrained(str(model_dir), device="cuda")
    try:
        await judge.judge(cases[0]["request"])                       # warm-up
        out = await asyncio.gather(*(judge.judge(c["request"]) for c in cases))
        report["vs_judge_fp32"] = compare(cases, out, ref)
        print("vs Judge fp32:", report["vs_judge_fp32"], flush=True)
        ticket = cases[0]["request"]
        report["burst"] = await burst(judge, ticket, 256)
        print("burst:", report["burst"], flush=True)
        if mode == "vllm":
            long = next(c for c in cases if c["name"] == "generated/long-state")["request"]
            first = await judge.judge(long)
            again = await judge.judge(long)
            salted = await judge.judge(long, cache_salt="another-tenant")
            report["cache"] = {"first": first["usage"]["cached_tokens"], "again": again["usage"]["cached_tokens"],
                               "other_salt": salted["usage"]["cached_tokens"],
                               "again_vs_first": max(abs(a - b) for a, b in
                                                     zip(flat(first["margins"]), flat(again["margins"])))}
            catalog = next(c for c in cases if c["name"] == "generated/intent-catalog-40")["request"]
            task = asyncio.create_task(judge.judge(dict(catalog, state="Отмена. " + catalog["state"])))
            await asyncio.sleep(0.005)
            task.cancel()
            try:
                await task
                cancelled = False
            except asyncio.CancelledError:
                cancelled = True
            after = await judge.judge(ticket)
            report["cancel"] = {"cancelled": cancelled, "next_request_answered": "answers" in after}
            print("cache:", report["cache"], "cancel:", report["cancel"], flush=True)
    finally:
        if mode == "vllm":
            judge.close()
        else:
            await judge.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["vllm", "torch"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    args = ap.parse_args()
    report = asyncio.run(check(args.mode, args))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"async_check_{args.mode}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
