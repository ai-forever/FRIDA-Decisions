"""Throughput of the async API: `AsyncJudge` (PyTorch) against `VllmJudge`, alternating runs.

    python tools/async_bench.py --razvilka path/to/razvilka/test.jsonl [--runs 5] [--gpu-memory-utilization 0.45]

Both judges are loaded in this process with their default settings (VllmJudge
with the given GPU memory share) and measured in turn, A B A B ..., after a
warm-up of 32 requests at once each. Every run starts cold: VllmJudge gets a
fresh `cache_salt` per run and `Judge`'s state cache is cleared, so no run
reads texts an earlier run left behind (texts shared inside a run still count:
that is what the cache is for). The texts each run took from a cache are
recorded with the rates:

* burst: 256 short tickets (distinct texts, ≈190 tokens) awaited at once;
* razvilka: all 735 items with 32 requests in flight at any time (a mix of
  lengths, median ≈22 state tokens).

Reports the median and the range of requests/s per workload and judge, and
writes `tests/_results/async_bench.json`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "benchmarks" / "razvilka")]
RESULTS = ROOT / "tests" / "_results"


def cold(judge):
    """A judge call that cannot read anything cached before this run."""
    from frida_decisions import AsyncJudge

    if isinstance(judge, AsyncJudge):
        if getattr(judge.wrapped, "state_cache", None) is not None:
            judge.wrapped.state_cache.clear()
        return judge.judge
    salt = f"run-{uuid.uuid4().hex}"
    return lambda request: judge.judge(request, cache_salt=salt)


def from_cache(responses) -> int:
    """vLLM: tokens read from the prefix cache; Judge: requests answered from its state cache."""
    return sum(r["usage"].get("cached_tokens", 0) or (r["usage"].get("state_cache") == "hit")
               for r in responses)


async def burst(judge, requests) -> tuple[float, int]:
    call = cold(judge)
    started = time.perf_counter()
    out = await asyncio.gather(*(call(r) for r in requests))
    return len(requests) / (time.perf_counter() - started), from_cache(out)


async def sustained(judge, requests, in_flight: int) -> tuple[float, int]:
    call = cold(judge)
    gate = asyncio.Semaphore(in_flight)

    async def one(r):
        async with gate:
            return await call(r)

    started = time.perf_counter()
    out = await asyncio.gather(*(one(r) for r in requests))
    return len(requests) / (time.perf_counter() - started), from_cache(out)


def summary(values: list[float]) -> dict:
    return {"median": round(statistics.median(values), 1), "min": round(min(values), 1),
            "max": round(max(values), 1), "runs": [round(v, 1) for v in values]}


async def main(args) -> dict:
    import conftest
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)      # conftest hides CUDA for the CPU tests
    import razvilka_eval as rz

    from frida_decisions import AsyncJudge, VllmJudge

    model_dir = str(conftest.MODEL_DIR)
    ticket = conftest.load_cases()[0]["request"]
    tickets = [dict(ticket, state=f"Обращение №{i}. " + ticket["state"]) for i in range(256)]
    warm = [dict(ticket, state=f"Прогрев №{i}. " + ticket["state"]) for i in range(32)]
    items = [it["request"] if isinstance(it["request"], dict) else json.loads(it["request"])
             for it in rz.load(args.razvilka)]

    vllm = VllmJudge.from_pretrained(model_dir, gpu_memory_utilization=args.gpu_memory_utilization)
    torch_judge = AsyncJudge.from_pretrained(model_dir, device="cuda")
    judges = {"AsyncJudge (PyTorch)": torch_judge, "VllmJudge": vllm}
    out = {name: {"burst": [], "razvilka": []} for name in judges}
    cached = {name: {"burst": [], "razvilka": []} for name in judges}
    try:
        for judge in judges.values():
            await burst(judge, warm)
            await sustained(judge, items[:64], args.in_flight)
        for run in range(args.runs):
            for name, judge in judges.items():
                rate, hits = await burst(judge, tickets)
                out[name]["burst"].append(rate)
                cached[name]["burst"].append(hits)
                print(f"run {run} burst    {name}: {rate:.1f} req/s, from cache {hits}", flush=True)
        for run in range(args.razvilka_runs):
            for name, judge in judges.items():
                rate, hits = await sustained(judge, items, args.in_flight)
                out[name]["razvilka"].append(rate)
                cached[name]["razvilka"].append(hits)
                print(f"run {run} razvilka {name}: {rate:.1f} req/s, from cache {hits}", flush=True)
    finally:
        vllm.close()
        await torch_judge.close()
    import torch
    report = {"gpu": torch.cuda.get_device_name(0), "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
              "in_flight": args.in_flight,
              "workloads": {"burst": "256 short tickets (≈190 tokens) awaited at once",
                            "razvilka": f"{len(items)} razvilka items, {args.in_flight} in flight"},
              "requests_per_second": {name: {w: summary(v) for w, v in res.items()} for name, res in out.items()},
              "from_cache_per_run": cached}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--razvilka", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--razvilka-runs", type=int, default=3)
    ap.add_argument("--in-flight", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    args = ap.parse_args()
    report = asyncio.run(main(args))
    print(json.dumps(report["requests_per_second"], ensure_ascii=False, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "async_bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
