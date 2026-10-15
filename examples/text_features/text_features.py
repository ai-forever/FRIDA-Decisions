"""Turn yes/no and score answers into features for a small logistic regression.

    python examples/text_features/text_features.py
    python examples/text_features/text_features.py --device cpu
    python examples/text_features/text_features.py --onnx
    python examples/text_features/text_features.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
try:
    from sklearn.linear_model import LogisticRegression
except ImportError as error:
    raise SystemExit("pip install scikit-learn") from error

FEATURES = ("angry", "demand", "concrete", "severity")
SEVERITY = [
    "Спокойный отзыв, делать ничего не нужно.",
    "Лёгкое недовольство, можно ответить шаблоном.",
    "Есть претензия, стоит прочитать.",
    "Явная претензия, лучше ответить человеку.",
    "Жёсткая претензия, откладывать нельзя.",
]


def request_for(text: str) -> dict:
    return {
        "state": text,
        "questions": {
            "angry": {"type": "noul", "instructions": "Автор раздражён?"},
            "demand": {"type": "noul", "instructions": "Автор требует вернуть деньги или что-то исправить?"},
            "concrete": {"type": "noul", "instructions": "Назван конкретный заказ, товар или дата?"},
            "severity": {"type": "score", "instructions": "Насколько отзыв требует реакции?",
                         "criteria": SEVERITY},
        },
    }


def featurize(answers: dict) -> list[float]:
    return [
        answers["angry"]["noul"],
        answers["demand"]["noul"],
        answers["concrete"]["noul"],
        answers["severity"]["score"] / 4.0,
    ]

# %% [markdown]
# ## The regression only sees the four numbers
#
# The split is fixed in `data.json`. Accuracy on eight test reviews is a sanity check. The coefficients show which question the linear model leaned on.

# %%
def run(judge, rows: list[dict]) -> None:
    responses = judge.judge_batch([request_for(row["text"]) for row in rows])
    matrix = [featurize(response["answers"]) for response in responses]
    train = [(x, row["escalate"]) for x, row in zip(matrix, rows) if row["split"] == "train"]
    test = [(x, row["escalate"]) for x, row in zip(matrix, rows) if row["split"] == "test"]
    model = LogisticRegression(max_iter=400)
    model.fit([x for x, _ in train], [y for _, y in train])
    predicted = model.predict([x for x, _ in test])
    gold = [y for _, y in test]
    hits = sum(int(left == right) for left, right in zip(predicted, gold))
    majority = max(set(y for _, y in train), key=lambda label: sum(y == label for _, y in train))
    baseline = sum(int(label == majority) for label in gold)
    print("features:", ", ".join(FEATURES))
    print("coefficients:", " ".join(f"{name}={coef:.2f}" for name, coef in zip(FEATURES, model.coef_[0])))
    print(f"test accuracy {hits}/{len(test)}")
    print(f"majority baseline (label {majority}) {baseline}/{len(test)}")
    for row, pred in zip((row for row in rows if row["split"] == "test"), predicted):
        mark = "ok" if pred == row["escalate"] else "miss"
        print(f"  {mark:4} pred {int(pred)} gold {row['escalate']}  {row['text'][:64]}")
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
