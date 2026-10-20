"""Decide whether two product cards are the same item.

    python examples/entity_alignment/entity_alignment.py
    python examples/entity_alignment/entity_alignment.py --device cpu
    python examples/entity_alignment/entity_alignment.py --onnx
    python examples/entity_alignment/entity_alignment.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

DATA = [
    {"gold": "same", "text": "Карточка А: электрочайник Philips, артикул HD9350, объём 1.7 литра. Карточка Б: чайник Philips модели HD9350 на 1.7 литра."},
    {"gold": "same", "text": "Карточка А: наушники Sony, артикул WH-1000XM5, цвет чёрный. Карточка Б: беспроводные наушники Sony WH-1000XM5, чёрные."},
    {"gold": "different", "text": "Карточка А: электрочайник Philips, артикул HD9350, объём 1.7 литра. Карточка Б: электрочайник Philips, артикул HD4649, объём 1.5 литра."},
    {"gold": "different", "text": "Карточка А: молоко «Домик в деревне» 3.2%, 1 литр. Карточка Б: стол IKEA Linnmon, белый, ширина 100 см."},
    {"gold": "different", "text": "Карточка А: стул IKEA Adde, белый. Карточка Б: стол IKEA Linnmon, белый, ширина 100 см."},
    {"gold": "different", "text": "Карточка А: чай Greenfield Golden Ceylon, 100 пакетиков. Карточка Б: чай Greenfield Flying Dragon, 100 пакетиков."},
]
SAME = [
    "Точно разные товары.",
    "Скорее разные товары.",
    "Неясно, тот же товар или другой.",
    "Скорее один и тот же товар.",
    "Точно один и тот же товар.",
]
SAME_MIN = 2.5
DIFFERENT_MAX = 1.5


def request_for(text):
    return {
        "state": text,
        "questions": {
            "same": {"type": "score", "instructions": "Это один и тот же товар в двух карточках?",
                     "criteria": SAME},
            "brand": {"type": "noul", "instructions": "Бренд в двух карточках один и тот же?"},
            "model": {"type": "noul", "instructions": "Артикул или модель в двух карточках совпадают?"},
        },
    }


def decide(answers):
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


def run(judge):
    hits = 0
    decided = 0
    for row in DATA:
        decision = decide(judge(request_for(row["text"]))["answers"])
        if decision["outcome"] == row["gold"]:
            mark = "ok"
            hits += 1
        else:
            mark = "miss"
        if decision["outcome"] != "review":
            decided += 1
        print(f"{mark:4} got {decision['outcome']:10} gold {row['gold']:10} "
              f"score {decision['score']:.2f}  brand {decision['brand']:.2f}  model {decision['model']:.2f}")
        print(f"     {row['text']}")
    print(f"\n{hits}/{len(DATA)} outcomes match the label, {decided}/{len(DATA)} decided without review")


def load_judge(args):
    if args.onnx:
        from frida_decisions import OnnxJudge
        return OnnxJudge.from_pretrained(args.model)
    from frida_decisions import Judge
    return Judge.from_pretrained(args.model, device=args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default=None, help="cuda or cpu (torch backend)")
    parser.add_argument("--onnx", action="store_true", help="int8 ONNX on CPU")

    args = parser.parse_args()
    run(load_judge(args))


if __name__ == "__main__":
    main()
