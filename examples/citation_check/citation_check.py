"""Decide whether a source supports a claim, contradicts it, or does not say.

    python examples/citation_check/citation_check.py
    python examples/citation_check/citation_check.py --device cpu
    python examples/citation_check/citation_check.py --onnx
    python examples/citation_check/citation_check.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
VERDICTS = {
    "supports": "источник подтверждает утверждение",
    "contradicts": "источник противоречит утверждению",
    "insufficient": "в источнике нет данных, чтобы судить",
}
# Below this, the choice is shown but a person decides. Several correct choices
# on this set sit around 0.3, so the cutoff is doing real work.
MIN_CONFIDENCE = 0.5


def request_for(source: str, claim: str) -> dict:
    return {
        "state": source,
        "questions": {"verdict": {
            "type": "choice",
            "instructions": "Утверждение: «" + claim + "». Как к нему относится источник?",
            "criteria": VERDICTS,
        }},
    }


def decide(answer: dict) -> dict:
    if answer["confidence"] < MIN_CONFIDENCE:
        return {"verdict": "review", "choice": answer["choice"], "confidence": answer["confidence"]}
    return {"verdict": answer["choice"], "choice": answer["choice"], "confidence": answer["confidence"]}

# %% [markdown]
# ## A third option for "the source does not say"
#
# `review` is not a model label. It is the branch taken when confidence is below the cutoff, so an uncertain `supports` is not stored as a fact.

# %%
def run(judge, rows: list[dict]) -> None:
    choice_hits = auto = auto_hits = 0
    for row in rows:
        response = judge(request_for(row["source"], row["claim"]))
        answer = response["answers"]["verdict"]
        decision = decide(answer)
        choice_ok = answer["choice"] == row["verdict"]
        choice_hits += choice_ok
        if decision["verdict"] == "review":
            mark = "review"
        else:
            auto += 1
            auto_hits += decision["verdict"] == row["verdict"]
            mark = "ok" if decision["verdict"] == row["verdict"] else "miss"
        print(f"{mark:6} choice {answer['choice']:14} gold {row['verdict']:14} "
              f"p={answer['confidence']:.2f}  action {decision['verdict']}")
        print(f"       {row['claim']}")
    print(f"\nchoice agrees with the label in {choice_hits}/{len(rows)}")
    if auto:
        print(f"auto at confidence >= {MIN_CONFIDENCE}: {auto}/{len(rows)}, "
              f"accuracy {auto_hits}/{auto}")
    else:
        print(f"nothing cleared confidence {MIN_CONFIDENCE}")
# --- end example ---


def load_rows():
    return json.loads((Path(__file__).parent / "data.json").read_text(encoding="utf-8"))


def load_judge(args):
    if args.onnx:
        from frida_decisions import OnnxJudge
        return OnnxJudge.from_pretrained(args.model, threads=args.threads)
    import torch

    from frida_decisions import Judge
    if args.threads:
        torch.set_num_threads(args.threads)
    return Judge.from_pretrained(args.model, device=args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default=None, help="cuda or cpu (torch backend)")
    parser.add_argument("--onnx", action="store_true", help="int8 ONNX on CPU")
    parser.add_argument("--threads", type=int, default=None)

    args = parser.parse_args()
    judge = load_judge(args)
    run(judge, load_rows())


if __name__ == "__main__":
    main()
