"""Score a letter on separate scales and combine the scores with weights in code.

    python examples/composite_scoring/composite_scoring.py
    python examples/composite_scoring/composite_scoring.py --device cpu
    python examples/composite_scoring/composite_scoring.py --onnx
    python examples/composite_scoring/composite_scoring.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
ROLE = "Вакансия: бэкенд на Python, нужны Django, PostgreSQL и опыт разбора чужих инцидентов."

RUBRICS = {
    "skills": [
        "Опыта по делу нет.",
        "Смежный опыт, нужного стека почти нет.",
        "Часть стека есть, без конкретных задач.",
        "Нужный стек и названы конкретные задачи.",
        "Стек, цифры и самостоятельные решения.",
    ],
    "clarity": [
        "Непонятно, что человек умеет.",
        "Общие слова, почти без фактов.",
        "Факты есть, но их приходится вычитывать.",
        "Письмо по делу и его легко дочитать.",
        "Коротко, по делу, с примерами.",
    ],
    "tone": [
        "Грубо или пренебрежительно.",
        "Фамильярно, рабочему письму не подходит.",
        "Нейтрально, без явных сбоев.",
        "Спокойно и уважительно.",
        "Уважительно и без заискивания.",
    ],
}

WEIGHTS = {
    "stack": {"skills": 0.7, "clarity": 0.2, "tone": 0.1},
    "writing": {"skills": 0.2, "clarity": 0.5, "tone": 0.3},
}
# `score` is the expected level, 0 .. 4. On this rubric the letters land in the
# lower half of the scale, so the invite cutoff is 1.15, not the midpoint.
CUTOFF = 1.15


def request_for(text: str) -> dict:
    return {
        "state": ROLE + "\n\nПисьмо кандидата:\n" + text,
        "questions": {
            "skills": {"type": "score", "instructions": "Насколько опыт совпадает с вакансией?",
                       "criteria": RUBRICS["skills"]},
            "clarity": {"type": "score", "instructions": "Насколько письмо ясное и конкретное?",
                        "criteria": RUBRICS["clarity"]},
            "tone": {"type": "score", "instructions": "Насколько тон уместен для рабочего письма?",
                     "criteria": RUBRICS["tone"]},
        },
    }


def weighted(answers: dict, weights: dict) -> float:
    return sum(weights[name] * answers[name]["score"] for name in weights)

# %% [markdown]
# ## Change a coefficient, not a prompt
#
# Both columns use the same three answers. Only the weights change, and with them who clears the cutoff.

# %%
def run(judge, rows: list[dict]) -> None:
    scored = []
    for row in rows:
        response = judge(request_for(row["text"]))
        answers = response["answers"]
        values = {name: weighted(answers, weights) for name, weights in WEIGHTS.items()}
        scored.append({**row, "answers": answers, "values": values})
        if response["usage"]["state_truncated"]:
            print(f"  {row['name']}: state truncated")

    print(f"{'name':10} {'skills':6} {'clarity':7} {'tone':6} {'stack':7} {'writing':7} invite")
    hits = 0
    for row in scored:
        answers = row["answers"]
        invite = row["values"]["stack"] >= CUTOFF
        hits += invite == row["invite"]
        print(f"{row['name']:10} {answers['skills']['score']:.2f}   {answers['clarity']['score']:.2f}    "
              f"{answers['tone']['score']:.2f}   {row['values']['stack']:.2f}    "
              f"{row['values']['writing']:.2f}    {str(invite):5} gold {row['invite']}")

    by_stack = [row["name"] for row in sorted(scored, key=lambda row: -row["values"]["stack"])]
    by_writing = [row["name"] for row in sorted(scored, key=lambda row: -row["values"]["writing"])]
    print(f"\norder by stack weights:   {', '.join(by_stack)}")
    print(f"order by writing weights: {', '.join(by_writing)}")
    print(f"cutoff {CUTOFF} on the stack weights agrees with the invite label in {hits}/{len(rows)}")
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
