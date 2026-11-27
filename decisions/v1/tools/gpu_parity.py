"""GPU check of the torch backend.

Runs every test case three ways and compares decisions and margins:

* cpu  — this package, float32 on CPU (the reference the CPU tests already pin);
* cuda — this package, bfloat16 on CUDA (the default GPU path);
* ref  — optional: the original research implementation on CUDA, same dtype
         (set FD_REFERENCE_JUDGE / FD_REFERENCE_CHECKPOINT as for the CPU test).

Also times a typical request on the GPU (median of 20 after warm-up).

    python tools/gpu_parity.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import torch  # noqa: E402

from frida_decisions import Judge  # noqa: E402
from frida_decisions.protocol import decision  # noqa: E402


def load_cases():
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)      # conftest hides CUDA for the CPU tests
    import conftest
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return conftest.load_cases(), conftest.MODEL_DIR


def flat(response):
    return [m for q in response["margins"].values() for m in q.values()]


def decisions(response):
    return {q: decision(a) for q, a in response["answers"].items()}


def run(judge, cases):
    return {c["name"]: judge.judge(c["request"]) for c in cases}


def compare(a, b):
    same = total = 0
    drift = 0.0
    for name in a:
        da, db = decisions(a[name]), decisions(b[name])
        total += len(da)
        same += sum(da[q] == db[q] for q in da)
        drift = max(drift, max(abs(x - y) for x, y in zip(flat(a[name]), flat(b[name]))))
    return {"same": same, "total": total, "max_margin_drift": round(drift, 6)}


def main() -> None:
    cases, model_dir = load_cases()
    assert torch.cuda.is_available(), "no CUDA device"
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(free - 1.5 * 2**30, 2**30) / total)

    cpu = Judge.from_pretrained(str(model_dir), device="cpu", dtype=torch.float32)
    out_cpu = run(cpu, cases)
    del cpu

    gpu = Judge.from_pretrained(str(model_dir), device="cuda")
    out_gpu = run(gpu, cases)
    report = {"cases": len(cases), "gpu": torch.cuda.get_device_name(0),
              "cuda_bf16_vs_cpu_fp32": compare(out_cpu, out_gpu)}

    typical = {"state": "Добрый день! " + "Пишу по поводу домашнего интернета, связь рвётся по вечерам. " * 24,
               "questions": {
                   "topic": {"type": "choice", "instructions": "Тема обращения?",
                             "criteria": {"internet": "домашний интернет", "mobile": "мобильная связь",
                                          "tv": "телевидение"}},
                   "refund": {"type": "noul", "instructions": "Клиент просит перерасчёт?"},
                   "mood": {"type": "score", "instructions": "Насколько клиент раздражён?",
                            "criteria": ["спокоен", "недоволен", "в ярости"]}}}
    for _ in range(3):
        gpu.judge(typical)
    times = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gpu.judge(typical); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    report["typical_request_ms_median"] = round(statistics.median(times), 1)
    report["typical_request_tokens"] = gpu.token_counts(typical) if hasattr(gpu, "token_counts") else None
    del gpu
    torch.cuda.empty_cache()

    ref_path, ckpt = os.environ.get("FD_REFERENCE_JUDGE"), os.environ.get("FD_REFERENCE_CHECKPOINT")
    if ref_path and ckpt:
        spec = importlib.util.spec_from_file_location("reference_judge", ref_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ref = module.Judge(ckpt, device="cuda", state_max=384)
        same = total = 0
        for c in cases:
            v = ref.judge(c["request"])
            ra = {q: decision(a) for q, a in v.answers.items()}
            ga = decisions(out_gpu[c["name"]])
            total += len(ra)
            same += sum(ra[q] == ga[q] for q in ra)
        report["cuda_package_vs_cuda_reference"] = {"same": same, "total": total}

    print(json.dumps(report, ensure_ascii=False, indent=1))
    out = ROOT / "tests" / "_results" / "gpu_parity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
