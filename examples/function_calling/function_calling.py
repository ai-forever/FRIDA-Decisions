"""Map an utterance to a typed function and ask again when confidence is low.

    python examples/function_calling/function_calling.py
    python examples/function_calling/function_calling.py --device cpu
    python examples/function_calling/function_calling.py --onnx
    python examples/function_calling/function_calling.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
import re

FUNCTIONS = {
    "weather": "прогноз погоды в названном городе",
    "tariff": "сменить тариф мобильной связи",
    "order": "узнать статус заказа по его номеру",
    "balance": "узнать остаток денег на счёте или карте",
}
CITY = {
    "msk": "Москва",
    "spb": "Санкт-Петербург",
    "kzn": "Казань",
    "sochi": "Сочи",
    "none": "город не назван",
}
PLAN = {
    "basic": "базовый тариф, мало интернета",
    "smart": "средний тариф с пакетом минут и гигабайт",
    "unlimited": "безлимитный интернет",
    "none": "тариф не назван",
}
PERIOD = {
    "month": "на один месяц",
    "year": "сразу на год",
    "none": "срок не назван",
}

MIN_CONFIDENCE = 0.55
ORDER_RE = re.compile(r"(?:заказ\w*|№)\s*№?\s*(\d{3,})", re.IGNORECASE)


def request_for(text: str) -> dict:
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


def weather(city: str) -> str:
    return f"Завтра в городе {CITY[city]} без осадков."


def change_tariff(plan: str, period: str) -> str:
    return f"Заявка: тариф {PLAN[plan]}, срок {PERIOD[period]}."


def order_status(number: str) -> str:
    return f"Заказ {number}: передан в доставку."


def balance() -> str:
    return "Остаток на счёте: 1 240 рублей."


def act(text: str, answers: dict) -> dict:
    fn = answers["fn"]
    if fn["confidence"] < MIN_CONFIDENCE:
        return {"outcome": "clarify", "ask": "Уточните: погода, тариф, статус заказа или баланс.",
                "confidence": fn["confidence"]}
    name = fn["choice"]
    if name == "weather":
        city = answers["city"]["choice"]
        if city == "none":
            return {"outcome": "clarify", "ask": "В каком городе посмотреть погоду?",
                    "confidence": fn["confidence"]}
        return {"outcome": f"weather:{city}", "result": weather(city), "confidence": fn["confidence"]}
    if name == "tariff":
        plan, period = answers["plan"]["choice"], answers["period"]["choice"]
        if plan == "none" or period == "none":
            return {"outcome": "clarify", "ask": "Какой тариф и на какой срок?",
                    "confidence": fn["confidence"]}
        return {"outcome": f"tariff:{plan}:{period}", "result": change_tariff(plan, period),
                "confidence": fn["confidence"]}
    if name == "order":
        found = ORDER_RE.search(text)
        if not found:
            return {"outcome": "clarify", "ask": "Назовите номер заказа.", "confidence": fn["confidence"]}
        return {"outcome": f"order:{found.group(1)}", "result": order_status(found.group(1)),
                "confidence": fn["confidence"]}
    return {"outcome": "balance", "result": balance(), "confidence": fn["confidence"]}

# %% [markdown]
# ## Code calls the function
#
# `outcome` is what the code did: a concrete call, or `clarify`. The gold column is the same shape, so the accuracy is the share of utterances where the code did the labelled thing.

# %%
def run(judge, rows: list[dict]) -> None:
    hits = 0
    for row in rows:
        response = judge(request_for(row["text"]))
        decision = act(row["text"], response["answers"])
        hits += decision["outcome"] == row["outcome"]
        detail = decision.get("result") or decision.get("ask")
        print(f"{'ok' if decision['outcome'] == row['outcome'] else 'miss':4} "
              f"got {decision['outcome']:22} gold {row['outcome']:22} "
              f"p={decision['confidence']:.2f}  {detail}")
    print(f"\n{hits}/{len(rows)} outcomes match the labels in data.json")
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
