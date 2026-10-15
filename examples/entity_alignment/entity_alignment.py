"""Decide whether two product cards are the same item, and which fields disagree.

    python examples/entity_alignment/entity_alignment.py
    python examples/entity_alignment/entity_alignment.py --device cpu
    python examples/entity_alignment/entity_alignment.py --onnx
    python examples/entity_alignment/entity_alignment.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
SAME = [
    "Точно разные товары.",
    "Скорее разные товары.",
    "Неясно, тот же товар или другой.",
    "Скорее один и тот же товар.",
    "Точно один и тот же товар.",
]
# score is 0..4. On prose cards a real match landed near 3.2 and a different
# model near 0.2, so 2.5 / 1.5 leave a band for "look at this pair".
SAME_MIN = 2.5
DIFFERENT_MAX = 1.5


def request_for(text: str) -> dict:
    return {
        "state": text,
        "questions": {
            "same": {"type": "score", "instructions": "Это один и тот же товар в двух карточках?",
                     "criteria": SAME},
            "brand": {"type": "noul", "instructions": "Бренд в двух карточках один и тот же?"},
            "model": {"type": "noul", "instructions": "Артикул или модель в двух карточках совпадают?"},
        },
    }


def decide(answers: dict) -> dict:
    """The score makes the call. The yes/no answers are reported, not vetoes:
    a field question can fire on a paraphrase even when the number matches.
    """
    score = answers["same"]["score"]
    if score >= SAME_MIN:
        outcome = "same"
    elif score < DIFFERENT_MAX:
        outcome = "different"
    else:
        outcome = "review"
    return {
        "outcome": outcome,
        "score": score,
        "brand": answers["brand"]["noul"],
        "model": answers["model"]["noul"],
    }

# %% [markdown]
# ## The score makes the call
#
# `same` at or above 2.5 is a match, below 1.5 is not, and the band between them waits for a person. The yes/no questions are printed beside the score. They are not a veto: a paraphrase can move them without moving the score.

# %%
def run(judge, rows: list[dict]) -> None:
    hits = decided = 0
    for row in rows:
        response = judge(request_for(row["text"]))
        decision = decide(response["answers"])
        match = decision["outcome"] == row["gold"]
        hits += match
        decided += decision["outcome"] != "review"
        print(f"{'ok' if match else 'miss':4} got {decision['outcome']:10} gold {row['gold']:10} "
              f"score {decision['score']:.2f}  brand {decision['brand']:.2f}  model {decision['model']:.2f}")
        print(f"     {row['text'].replace(chr(10), ' | ')}")
    print(f"\n{hits}/{len(rows)} outcomes match the label, {decided}/{len(rows)} decided without review")
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
