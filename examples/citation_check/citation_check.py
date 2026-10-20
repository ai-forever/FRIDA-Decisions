"""Decide whether a source supports a claim, contradicts it, or does not say.

    python examples/citation_check/citation_check.py
    python examples/citation_check/citation_check.py --device cpu
    python examples/citation_check/citation_check.py --onnx
    python examples/citation_check/citation_check.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

DATA = [
    {"source": "Возврат товара надлежащего качества возможен в течение 14 дней, если сохранены товарный вид и чек.",
     "claim": "Качественный товар можно вернуть в течение 14 дней при наличии чека.",
     "verdict": "supports"},
    {"source": "Возврат товара надлежащего качества возможен в течение 14 дней, если сохранены товарный вид и чек.",
     "claim": "Вернуть качественный товар можно в любой срок и без чека.",
     "verdict": "contradicts"},
    {"source": "Возврат товара надлежащего качества возможен в течение 14 дней, если сохранены товарный вид и чек.",
     "claim": "Доставка по городу бесплатная.",
     "verdict": "insufficient"},
    {"source": "Курс обмена сдаётся до 18:00, после этого заявки переносятся на следующий рабочий день.",
     "claim": "Заявку на обмен после 18:00 рассмотрят на следующий рабочий день.",
     "verdict": "supports"},
    {"source": "Курс обмена сдаётся до 18:00, после этого заявки переносятся на следующий рабочий день.",
     "claim": "Обмен работает круглосуточно, включая выходные.",
     "verdict": "contradicts"},
    {"source": "Гарантия на чайник — один год. Сколы колбы от удара гарантия не покрывает.",
     "claim": "Если колба треснула от удара, гарантия это покрывает.",
     "verdict": "contradicts"},
    {"source": "Гарантия на чайник — один год. Сколы колбы от удара гарантия не покрывает.",
     "claim": "Гарантия на чайник длится один год.",
     "verdict": "supports"},
    {"source": "Гарантия на чайник — один год. Сколы колбы от удара гарантия не покрывает.",
     "claim": "В комплекте есть запасной фильтр.",
     "verdict": "insufficient"},
]
VERDICTS = {
    "supports": "источник подтверждает утверждение",
    "contradicts": "источник противоречит утверждению",
    "insufficient": "в источнике нет данных, чтобы судить",
}
MIN_CONFIDENCE = 0.5


def request_for(source, claim):
    return {
        "state": source,
        "questions": {"verdict": {
            "type": "choice",
            "instructions": "Утверждение: «" + claim + "». Как к нему относится источник?",
            "criteria": VERDICTS,
        }},
    }


def decide(answer):
    if answer["confidence"] < MIN_CONFIDENCE:
        action = "review"
    else:
        action = answer["choice"]
    return {"action": action, "choice": answer["choice"], "confidence": answer["confidence"]}


def run(judge):
    choice_hits = 0
    auto = 0
    auto_hits = 0
    for row in DATA:
        answer = judge(request_for(row["source"], row["claim"]))["answers"]["verdict"]
        decision = decide(answer)
        if answer["choice"] == row["verdict"]:
            choice_hits += 1
        if decision["action"] == "review":
            mark = "review"
        elif decision["action"] == row["verdict"]:
            mark = "ok"
            auto += 1
            auto_hits += 1
        else:
            mark = "miss"
            auto += 1
        print(f"{mark:6} choice {answer['choice']:14} gold {row['verdict']:14} "
              f"p={answer['confidence']:.2f}  action {decision['action']}")
        print(f"       {row['claim']}")
    print(f"\nchoice agrees with the label in {choice_hits}/{len(DATA)}")
    if auto:
        print(f"auto at confidence >= {MIN_CONFIDENCE}: {auto}/{len(DATA)}, accuracy {auto_hits}/{auto}")
    else:
        print(f"nothing cleared confidence {MIN_CONFIDENCE}")


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
