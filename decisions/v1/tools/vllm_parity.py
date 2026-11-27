"""Check a running vLLM server against `Judge` (float32, CPU).

    python tools/vllm_parity.py --reference-only     # before starting the server
    python tools/vllm_parity.py --url http://127.0.0.1:8000

Every test case is sent twice to `POST /pooling`. The first pass meets an empty
state cache (rows of one request still share their state within a step); the
second pass finds every state cached. For each pass: decisions and the largest
margin difference against `Judge`, and the cached tokens the server reports.
Then the same rows as raw token ids (`"task": "classify"`), and requests that
must be refused (an invalid request, a row whose state hash is forged) without
taking the server down.

The reference is computed on CPU in float32 and kept in
`tests/_results/vllm_reference_fp32.json`. Compute it before starting a server
in Docker on Windows: a container loading weights through a bind mount hangs
while a Windows process holds the same file memory-mapped.

Writes `tests/_results/vllm_parity.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

RESULTS = ROOT / "tests" / "_results"
REFERENCE = RESULTS / "vllm_reference_fp32.json"


def load_cases():
    import conftest
    return conftest.load_cases(), conftest.MODEL_DIR


def flat(margins: dict) -> list[float]:
    return [m for per_q in margins.values() for m in per_q.values()]


def reference(cases, model_dir) -> dict:
    import torch

    from frida_decisions import Judge

    judge = Judge.from_pretrained(str(model_dir), device="cpu", dtype=torch.float32)
    out = {}
    for case in cases:
        response = judge.judge(case["request"])
        out[case["name"]] = {"request": case["request"], "margins": response["margins"],
                             "answers": response["answers"]}
    del judge
    RESULTS.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("FD_VLLM_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()

    cases, model_dir = load_cases()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() else None
    if ref is None or {c["name"] for c in cases} - set(ref) or any(
            ref[c["name"]]["request"] != c["request"] or "answers" not in ref[c["name"]] for c in cases):
        ref = reference(cases, model_dir)
    if args.reference_only:
        print(f"wrote {REFERENCE}")
        return

    import httpx

    from frida_decisions.base import BaseJudge
    from frida_decisions.protocol import aggregate, decision
    from frida_decisions.vllm_backend.rows import PREFIX, build_rows, tokenize_rows

    client = httpx.Client(base_url=args.url, timeout=600)
    model = client.get("/v1/models").json()["data"][0]["id"]
    text = BaseJudge(model_dir, state_max=384)
    vocab = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))["vocab_size"]

    def ask(request: dict) -> dict:
        r = client.post("/pooling", json={"model": model, "data": request})
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
        return r.json()["data"]

    report = {"url": args.url, "model": model, "cases": len(cases)}
    for p in range(2):
        same = total = cached = prompt = 0
        drift = 0.0
        per_case = {}
        for case in cases:
            data = ask(case["request"])
            ours, theirs = flat(data["margins"]), flat(ref[case["name"]]["margins"])
            parsed, candidates, _ = text.compile(case["request"])
            a = {q: decision(x) for q, x in aggregate(parsed, candidates, theirs).items()}
            b = {q: decision(x) for q, x in data["answers"].items()}
            same += sum(a[q] == b[q] for q in a)
            total += len(a)
            d = max(abs(x - y) for x, y in zip(ours, theirs))
            drift = max(drift, d)
            cached += data["usage"]["cached_tokens"]
            prompt += data["usage"]["prompt_tokens"]
            per_case[case["name"]] = {"max_margin_drift": d, "cached_tokens": data["usage"]["cached_tokens"],
                                      "rows": data["usage"]["rows"]}
        report[f"pass{p}"] = {"same_decisions": f"{same}/{total}", "max_margin_drift": drift,
                              "cached_tokens": cached, "prompt_tokens": prompt, "per_case": per_case}
        print(f"pass {p}: {same}/{total} same decisions, max |dmargin| {drift:.3g}, "
              f"cached {cached} of {prompt} prompt tokens", flush=True)

    # The same rows as raw ids, against the request path of the same server.
    raw = 0.0
    for case in cases:
        _, candidates, _ = text.compile(case["request"])
        tok, _ = tokenize_rows(candidates, text.text, text.config, text.state_max)
        rows = build_rows(tok, text.config, vocab)
        r = client.post("/pooling", json={"model": model, "task": "classify",
                                          "input": [row.ids for row in rows]})
        r.raise_for_status()
        margins = [math.nan] * len(candidates)
        for row, item in zip(rows, r.json()["data"]):
            for i, value in zip(row.candidates, item["data"]):
                margins[i] = value
        raw = max(raw, max(abs(x - y) for x, y in zip(margins, flat(ref[case["name"]]["margins"]))))
    report["raw_ids_max_margin_drift"] = raw
    print(f"raw ids: max |dmargin| {raw:.3g}", flush=True)

    # Refusals: an invalid request, and a row whose state hash is forged.
    bad = client.post("/pooling", json={"model": model, "data": {"state": "x", "questions": {}}})
    _, candidates, _ = text.compile(cases[0]["request"])
    tok, _ = tokenize_rows(candidates, text.text, text.config, text.state_max)
    row = build_rows(tok, text.config, vocab)[0].ids
    forged = client.post("/pooling", json={"model": model, "task": "classify",
                                           "input": [[row[0], 77, 78, 79, 80] + row[PREFIX:]]})
    report["refusals"] = {"invalid_request": bad.status_code, "forged_state_hash": forged.status_code,
                          "health_after": client.get("/health").status_code}
    print(f"refusals: {report['refusals']}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "vllm_parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
