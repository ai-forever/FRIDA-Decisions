"""Route a support ticket only when the team choice is confident enough.

    python examples/support_triage/support_triage.py
    python examples/support_triage/support_triage.py --device cpu
    python examples/support_triage/support_triage.py --onnx
    python examples/support_triage/support_triage.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
TEAMS = {
    "auth": "доступ к аккаунту, вход, пароль, личные данные",
    "billing": "оплата, списания, возвраты денег, подписки",
    "shipping": "доставка, курьер, сроки и адрес получения заказа",
    "product": "качество, комплектность и характеристики товара",
}

URGENCY = [
    "Очень низкая: работа не страдает, ответить можно когда угодно.",
    "Низкая: неудобство небольшое, ответить можно на этой неделе.",
    "Средняя: работа затруднена, но обходной путь есть.",
    "Высокая: работа заблокирована, обходного пути нет.",
    "Критическая: потери растут с каждым часом.",
]

# Starting point, not a measured constant. The sweep below is the measurement.
DEFAULT_CONFIDENCE = 0.7
HIGH_URGENCY = 3.0
UPSET = 0.6


def request_for(text: str) -> dict:
    return {
        "state": text,
        "questions": {
            "team": {"type": "choice", "instructions": "В какую команду направить обращение?",
                     "criteria": TEAMS},
            "urgency": {"type": "score", "instructions": "Оцени срочность обращения по шкале ниже.",
                        "criteria": URGENCY},
            "angry": {"type": "noul", "instructions": "Автор раздражён?"},
            "churn": {"type": "noul", "instructions": "Автор грозит уйти или больше не покупать?"},
        },
    }


def route(answers: dict, min_confidence: float) -> dict:
    """Team queue when the choice is confident enough, otherwise a person.

    Urgency and the two yes/no checks only set the priority. They do not pick the team.
    """
    team = answers["team"]
    urgent = answers["urgency"]["score"] >= HIGH_URGENCY
    upset = answers["angry"]["noul"] >= UPSET or answers["churn"]["noul"] >= UPSET
    auto = team["confidence"] >= min_confidence
    return {
        "queue": team["choice"] if auto else "operator",
        "auto": auto,
        "priority": "high" if urgent or upset else "normal",
        "confidence": team["confidence"],
    }


def evaluate(scored: list[dict], min_confidence: float) -> dict:
    decided = []
    correct = 0
    for row in scored:
        decision = route(row["answers"], min_confidence)
        if not decision["auto"]:
            continue
        decided.append(decision)
        correct += decision["queue"] == row["team"]
    return {
        "n": len(scored),
        "auto": len(decided),
        "coverage": len(decided) / len(scored),
        "accuracy": (correct / len(decided)) if decided else None,
    }

# %% [markdown]
# ## The threshold lives in code
#
# Every ticket is scored once. Coverage is the share sent to a team without a person.
# Accuracy is only over those tickets: a ticket handed to an operator is not counted as a miss.

# %%
def run(judge, rows: list[dict]) -> None:
    responses = judge.judge_batch([request_for(row["text"]) for row in rows])
    scored = []
    print(f"{'result':8} {'queue':10} {'pri':6} conf  urgency angry churn  gold")
    for row, response in zip(rows, responses):
        decision = route(response["answers"], DEFAULT_CONFIDENCE)
        answers = response["answers"]
        scored.append({"team": row["team"], "answers": answers})
        if decision["auto"] and decision["queue"] == row["team"]:
            mark = "ok"
        elif decision["auto"]:
            mark = "miss"
        else:
            mark = "hold"
        print(f"{mark:8} {decision['queue']:10} {decision['priority']:6} "
              f"{decision['confidence']:.2f}  {answers['urgency']['score']:.2f}    "
              f"{answers['angry']['noul']:.2f}  {answers['churn']['noul']:.2f}   {row['team']}")
        if response["usage"]["state_truncated"]:
            print("  state truncated: trained range is 512 tokens, the default cut is 384")

    print(f"\nthreshold {DEFAULT_CONFIDENCE}:")
    summary = evaluate(scored, DEFAULT_CONFIDENCE)
    accuracy = "n/a" if summary["accuracy"] is None else f"{summary['accuracy']:.0%}"
    print(f"  auto {summary['auto']}/{summary['n']} ({summary['coverage']:.0%}), "
          f"accuracy on auto {accuracy}")
    print("\nthreshold sweep (accuracy is only over tickets the model routes itself):")
    for threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        summary = evaluate(scored, threshold)
        accuracy = "n/a" if summary["accuracy"] is None else f"{summary['accuracy']:.0%}"
        print(f"  {threshold:.2f}  coverage {summary['coverage']:.0%}  accuracy {accuracy}")
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
