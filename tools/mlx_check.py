"""Check MlxJudge, and PyTorch on MPS, on an Apple Silicon Mac.

    pip install -e ".[dev,mlx]" datasets
    python tools/mlx_check.py [--model DIR_OR_REPO] [--limit N] [--skip-torch-cpu]
                              [--no-latency] [--out tests/_results/mlx_check.json]

All backends score the same razvilka items with the state cut at 512, as
tools/release_eval.py does:

  1. PyTorch CPU float32: the reference;
  2. MLX float32 and bfloat16;
  3. PyTorch on MPS (bfloat16, the package default on a non-CUDA GPU).

For each: accuracy, decisions equal to the reference and margin drift. Then
the 243-intent catalog through MLX packed, cached (miss) and cached (hit)
against the reference; MlxJudge under AsyncJudge (its worker thread) against
the same calls made directly; latency of the ≈400-token request of
tools/release_eval.py (keys "384_tokens_*", as there) with 1 and 3 questions
and of the catalog (medians of 30 and 10 calls after 3 warm-up calls); peak
MLX memory.

The JSON is rewritten after every stage, so a run cut short still leaves
what it measured.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks" / "razvilka")]

import razvilka_eval as rz  # noqa: E402

STATE_MAX = 512

p = argparse.ArgumentParser()
p.add_argument("--model", default="ai-forever/FRIDA-Decisions")
p.add_argument("--revision", default=None)
p.add_argument("--data", default=None, help="local razvilka test.jsonl (default: the Hub)")
p.add_argument("--limit", type=int, default=None, help="first N razvilka items only")
p.add_argument("--skip-torch-cpu", action="store_true",
               help="no CPU reference: decisions and margins are compared with MLX float32")
p.add_argument("--skip-mps", action="store_true")
p.add_argument("--runs", type=int, default=30)
p.add_argument("--no-latency", action="store_true",
               help="parity only (for MLX's CPU build, where timings mean nothing)")
p.add_argument("--out", default=str(ROOT / "tests" / "_results" / "mlx_check.json"))
ARGS = p.parse_args()
REPORT: dict = {}


def save() -> None:
    Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ARGS.out).write_text(json.dumps(REPORT, ensure_ascii=False, indent=1, default=str),
                              encoding="utf-8")


def sysctl(name: str):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    import mlx.core as mx

    env = {"machine": platform.machine(), "macos": platform.mac_ver()[0] or platform.platform(),
           "python": platform.python_version(), "chip": sysctl("machdep.cpu.brand_string"),
           "model": sysctl("hw.model"), "memory_gib": None,
           "mlx": getattr(mx, "__version__", None), "mlx_device": str(mx.default_device())}
    mem = sysctl("hw.memsize")
    if mem and mem.isdigit():
        env["memory_gib"] = round(int(mem) / 2**30)
    try:
        import torch
        env["torch"] = torch.__version__
        env["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        env["torch"] = None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
        env["commit"] = out.stdout.strip() or None
    except Exception:
        env["commit"] = None
    return env


# ------------------------------------------------------------------ helpers
def run_razvilka(judge, items) -> tuple[dict, dict, float]:
    """Answers and per-option margins of every item, and the wall time."""
    answers, margins = {}, {}
    t0 = time.perf_counter()
    for item in items:
        request = item["request"] if isinstance(item["request"], dict) else json.loads(item["request"])
        (qid, _), = request["questions"].items()
        out = judge.judge(request)
        answers[item["id"]] = out["answers"][qid]
        margins[item["id"]] = out["margins"][qid]
    return answers, margins, time.perf_counter() - t0


def drift(a: dict, b: dict) -> dict:
    """Absolute margin differences over every option of every item."""
    diffs = sorted(abs(a[i][o] - b[i][o]) for i in a for o in a[i])
    q = lambda f: round(diffs[min(len(diffs) - 1, int(f * len(diffs)))], 6)  # noqa: E731
    return {"options": len(diffs), "max": round(diffs[-1], 6), "median": q(0.5),
            "p95": q(0.95), "p99": q(0.99)}


def compare(name: str, answers, margins, seconds, items, ref) -> dict:
    score = rz.score(answers, items)
    out = {"accuracy": score["accuracy"], "correct": score["correct"],
           "per_type": score["per_type"], "minutes": round(seconds / 60, 2)}
    if ref is not None and ref["name"] != name:
        differ = [i["id"] for i in items
                  if rz.decide(i, answers[i["id"]]) != rz.decide(i, ref["answers"][i["id"]])]
        out["reference"] = ref["name"]
        out["decisions_equal"] = len(items) - len(differ)
        out["different"] = differ
        out["margin_drift"] = drift(margins, ref["margins"])
    return out


def typical(questions: int, i: int) -> dict:
    """The 384-token request of tools/release_eval.py."""
    state = f"Обращение {i}. " + "Пишу по поводу домашнего интернета, связь рвётся по вечерам, прошу перерасчёт. " * 22
    qs = {"topic": {"type": "choice", "instructions": "Тема обращения?",
                    "criteria": {"internet": "домашний интернет", "mobile": "мобильная связь", "tv": "телевидение"}},
          "refund": {"type": "noul", "instructions": "Клиент просит перерасчёт?"},
          "mood": {"type": "score", "instructions": "Насколько клиент раздражён?",
                   "criteria": ["спокоен", "недоволен", "в ярости"]}}
    return {"state": state, "questions": dict(list(qs.items())[:questions])}


def median_ms(call, runs: int) -> float:
    """Every backend returns Python floats, so a call ends after the device work."""
    for i in range(3):
        call(-1 - i)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        call(i)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1)


def latency(judge, catalog: dict) -> dict:
    def catalog_cold(_):
        if judge.state_cache is not None:
            judge.state_cache.clear()
        judge.judge(catalog)

    return {"384_tokens_1_question": median_ms(lambda i: judge.judge(typical(1, i)), ARGS.runs),
            "384_tokens_3_questions": median_ms(lambda i: judge.judge(typical(3, i)), ARGS.runs),
            "catalog_243_cold": median_ms(catalog_cold, max(5, ARGS.runs // 3)),
            "catalog_243_cache_hit": median_ms(lambda _: judge.judge(catalog), max(5, ARGS.runs // 3)),
            "tokens_3q": judge.token_counts(typical(3, 0))}


def mlx_peak(reset: bool = False):
    import mlx.core as mx

    get = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    if reset:
        (getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory)()
        return None
    return round(get() / 2**30, 3)


def stage(name: str, fn) -> None:
    print(f"\n== {name}", flush=True)
    t0 = time.perf_counter()
    try:
        REPORT[name] = fn()
    except Exception:
        REPORT[name] = {"error": traceback.format_exc()}
        print(REPORT[name]["error"], flush=True)
    print(f"   {time.perf_counter() - t0:.0f} s", flush=True)
    save()


# ------------------------------------------------------------------ main
def main() -> None:
    REPORT["environment"] = environment()
    REPORT["args"] = vars(ARGS)
    print(json.dumps(REPORT["environment"], indent=1))
    if REPORT["environment"]["machine"] != "arm64":
        print("Not Apple Silicon: MLX runs on its CPU build only; Metal is not checked.")
    save()

    import mlx.core as mx

    from frida_decisions import MlxJudge
    from frida_decisions.base import resolve_model_dir

    folder = resolve_model_dir(ARGS.model, ["*.json", "model.safetensors", "head.safetensors"],
                               ARGS.revision)
    items = rz.load(ARGS.data) if ARGS.data else rz.load()
    if ARGS.limit:
        items = items[:ARGS.limit]
    REPORT["items"] = len(items)
    catalog_file = json.loads((ROOT / "examples" / "data" / "intent_catalog.json").read_text("utf-8"))
    catalog = catalog_file["request"]
    ref = None

    if not ARGS.skip_torch_cpu:
        def torch_cpu():
            nonlocal ref
            from frida_decisions import Judge
            judge = Judge.from_pretrained(folder, device="cpu", state_max=STATE_MAX)
            answers, margins, seconds = run_razvilka(judge, items)
            ref = {"name": "torch_cpu_fp32", "answers": answers, "margins": margins}
            cat = judge.judge(catalog)
            ref["catalog"] = cat["margins"]["intent"]
            out = compare("torch_cpu_fp32", answers, margins, seconds, items, None)
            out["catalog_choice"] = cat["answers"]["intent"]["choice"]
            return out
        stage("torch_cpu_fp32", torch_cpu)

    for dtype_name, dtype in (("fp32", mx.float32), ("bf16", mx.bfloat16)):
        name = f"mlx_{dtype_name}"

        def run_mlx(dtype=dtype, name=name):
            nonlocal ref
            mlx_peak(reset=True)
            t0 = time.perf_counter()
            judge = MlxJudge.from_pretrained(folder, dtype=dtype, state_max=STATE_MAX)
            load_s = time.perf_counter() - t0
            answers, margins, seconds = run_razvilka(judge, items)
            if ref is None:             # --skip-torch-cpu: MLX float32 is the reference
                ref = {"name": name, "answers": answers, "margins": margins}
            out = compare(name, answers, margins, seconds, items, ref)
            out["load_seconds"] = round(load_s, 2)

            # The catalog needs 16 rows, so it takes the cached path; check all three ways.
            judge.state_cache.clear()
            packed = judge.margins_packed(catalog)
            miss = judge.margins_cached(catalog)
            hit = judge.margins_cached(catalog)
            options = list(judge.judge(catalog)["margins"]["intent"])
            as_map = lambda values: {"c": dict(zip(options, values))}  # noqa: E731
            out["catalog"] = {"cached_vs_packed": drift(as_map(miss), as_map(packed)),
                              "hit_vs_miss": drift(as_map(hit), as_map(miss)),
                              "choice": judge.judge(catalog)["answers"]["intent"]["choice"],
                              "expected": catalog_file["expected"]["intent"]}
            if ref.get("catalog"):
                out["catalog"]["vs_reference"] = drift(as_map(miss), {"c": ref["catalog"]})
            if not ARGS.no_latency:
                out["latency_ms"] = latency(judge, catalog)
            out["peak_mlx_gib"] = mlx_peak()
            return out
        stage(name, run_mlx)

    def run_async():
        """AsyncJudge (0.3.0) scores in a worker thread; the model is loaded in this one."""
        import asyncio

        try:
            from frida_decisions import AsyncJudge
        except ImportError:
            return {"skipped": "no AsyncJudge in this checkout (before 0.3.0)"}
        judge = MlxJudge.from_pretrained(folder, state_max=STATE_MAX)
        requests = [typical(3, i) for i in range(12)] + [catalog, catalog]
        sync = [judge.judge(r)["margins"] for r in requests]
        judge.state_cache.clear()

        async def gather():
            async with AsyncJudge(judge) as wrapped:
                return await asyncio.gather(*(wrapped.judge(r) for r in requests))

        out = asyncio.run(gather())
        worst = max(abs(o["margins"][q][k] - s[q][k])
                    for o, s in zip(out, sync) for q in s for k in s[q])
        return {"requests": len(requests), "max_margin_drift_vs_sync": round(worst, 6),
                "backend": out[0]["usage"]["backend"]}
    stage("async_mlx_fp32", run_async)

    if not ARGS.skip_mps:
        def run_mps(dtype_name: str):
            import torch
            from frida_decisions import Judge
            if not torch.backends.mps.is_available():
                return {"skipped": "MPS not available"}
            dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
            judge = Judge.from_pretrained(folder, device="mps", dtype=dtype, state_max=STATE_MAX)
            out = {}
            if dtype_name == "bf16":    # the package default on a non-CUDA GPU
                answers, margins, seconds = run_razvilka(judge, items)
                out = compare(f"torch_mps_{dtype_name}", answers, margins, seconds, items, ref)
            if not ARGS.no_latency:
                out["latency_ms"] = latency(judge, catalog)
            return out
        stage("torch_mps_bf16", lambda: run_mps("bf16"))
        stage("torch_mps_fp32_latency", lambda: run_mps("fp32"))

    print(f"\nDone. Results: {Path(ARGS.out).resolve()}")


if __name__ == "__main__":
    main()
