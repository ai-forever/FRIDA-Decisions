"""Map an utterance to a typed function and ask again when confidence is low.

    python examples/function_calling/function_calling.py
    python examples/function_calling/function_calling.py --device cpu
    python examples/function_calling/function_calling.py --onnx
    python examples/function_calling/function_calling.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

import re

DATA = [
    {"text": "Какая завтра погода в Казани?", "outcome": "weather:kzn"},
    {"text": "Переведите меня на безлимитный тариф на год.", "outcome": "tariff:unlimited:year"},
    {"text": "Где мой заказ №5512?", "outcome": "order:5512"},
    {"text": "Сколько денег на карте?", "outcome": "balance"},
    {"text": "Хочу сменить тариф.", "outcome": "clarify"},
    {"text": "Погода.", "outcome": "clarify"},
    {"text": "Проверьте статус заказа 8841.", "outcome": "order:8841"},
    {"text": "Баланс счёта, пожалуйста, кредит не предлагайте.", "outcome": "balance"},
]
FUNCTIONS = {
    "weather": "прогноз погоды в названном городе",
    "tariff": "сменить тариф мобильной связи",
    "order": "узнать статус заказа по его номеру",
    "balance": "узнать остаток денег на счёте или карте",
}
CITY = {
    "msk": "Москва", "spb": "Санкт-Петербург", "kzn": "Казань", "sochi": "Сочи",
    "none": "город не назван",
}
PLAN = {
    "basic": "базовый тариф, мало интернета",
    "smart": "средний тариф с пакетом минут и гигабайт",
    "unlimited": "безлимитный интернет",
    "none": "тариф не назван",
}
PERIOD = {
    "month": "на один месяц", "year": "сразу на год", "none": "срок не назван",
}
MIN_CONFIDENCE = 0.55
ORDER_RE = re.compile(r"(?:заказ\w*|№)\s*№?\s*(\d{3,})", re.IGNORECASE)


def request_for(text):
    return {
        "state": text,
        "questions": {
            "fn": {"type": "choice", "instructions": "Какую функцию вызвать?", "criteria": FUNCTIONS},
            "city": {"type": "choice", "instructions": "О каком городе речь? Если город не назван, выбери none.",
                     "criteria": CITY},
            "plan": {"type": "choice", "instructions": "Какой тариф назван? Если не назван, выбери none.",
                     "criteria": PLAN},
            "period": {"type": "choice", "instructions": "На какой срок менять тариф? Если срок не назван, выбери none.",
                       "criteria": PERIOD},
        },
    }


def weather(city):
    return f"Завтра в городе {CITY[city]} без осадков."


def change_tariff(plan, period):
    return f"Заявка: тариф {PLAN[plan]}, срок {PERIOD[period]}."


def order_status(number):
    return f"Заказ {number}: передан в доставку."


def balance():
    return "Остаток на счёте: 1 240 рублей."


def decide(text, answers):
    fn = answers["fn"]
    if fn["confidence"] < MIN_CONFIDENCE:
        return {"outcome": "clarify", "detail": "Уточните: погода, тариф, статус заказа или баланс.",
                "confidence": fn["confidence"]}
    name = fn["choice"]
    if name == "weather":
        city = answers["city"]["choice"]
        if city == "none":
            return {"outcome": "clarify", "detail": "В каком городе посмотреть погоду?",
                    "confidence": fn["confidence"]}
        return {"outcome": f"weather:{city}", "detail": weather(city), "confidence": fn["confidence"]}
    if name == "tariff":
        plan = answers["plan"]["choice"]
        period = answers["period"]["choice"]
        if plan == "none" or period == "none":
            return {"outcome": "clarify", "detail": "Какой тариф и на какой срок?",
                    "confidence": fn["confidence"]}
        return {"outcome": f"tariff:{plan}:{period}", "detail": change_tariff(plan, period),
                "confidence": fn["confidence"]}
    if name == "order":
        found = ORDER_RE.search(text)
        if not found:
            return {"outcome": "clarify", "detail": "Назовите номер заказа.",
                    "confidence": fn["confidence"]}
        return {"outcome": f"order:{found.group(1)}", "detail": order_status(found.group(1)),
                "confidence": fn["confidence"]}
    return {"outcome": "balance", "detail": balance(), "confidence": fn["confidence"]}


def run(judge):
    hits = 0
    for row in DATA:
        decision = decide(row["text"], judge(request_for(row["text"]))["answers"])
        if decision["outcome"] == row["outcome"]:
            mark = "ok"
            hits += 1
        else:
            mark = "miss"
        print(f"{mark:4} got {decision['outcome']:22} gold {row['outcome']:22} "
              f"p={decision['confidence']:.2f}  {decision['detail']}")
    print(f"\n{hits}/{len(DATA)} outcomes match the labels")


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
