"""Let a regex propose spans and a choice question pick the one that was asked for.

    python examples/value_extraction/value_extraction.py
    python examples/value_extraction/value_extraction.py --device cpu
    python examples/value_extraction/value_extraction.py --onnx
    python examples/value_extraction/value_extraction.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

import re

DATA = [
    {"kind": "money", "value": "500",
     "text": "К возврату 500 рублей. Доставка стоила 300 рублей и не возвращается."},
    {"kind": "money", "value": "2490",
     "text": "К возврату 2 490 рублей за сам заказ, доставка 300 рублей не возвращается."},
    {"kind": "phone", "value": "79160000000",
     "text": "Рабочий телефон 8 916 000-00-00. Старый номер 8 903 111-22-33 отключён."},
    {"kind": "phone", "value": "79990001122",
     "text": "Пишите только на +7 999 000-11-22. Номер 8 916 000-00-00 больше не обслуживается."},
    {"kind": "date", "value": "15 марта",
     "text": "Заказ оформили 2 марта, курьер привезёт его 15 марта."},
    {"kind": "date", "value": "3 апреля 2026",
     "text": "Оплатили 1 апреля 2026, доставка назначена на 3 апреля 2026."},
    {"kind": "money", "value": "none",
     "text": "Возврат оформили, точную сумму назовут после проверки склада."},
    {"kind": "phone", "value": "none",
     "text": "Перезвоните, когда будет минута, номер я пришлю отдельно."},
]
PATTERNS = {
    "phone": re.compile(r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
    "money": re.compile(r"\d[\d\s]*\d\s*(?:руб(?:лей|ля|\.)?|₽)"),
    "date": re.compile(
        r"\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
        r"сентября|октября|ноября|декабря)(?:\s+\d{4})?"
    ),
}
ASK = {
    "phone": "Какая из строк — телефон, на который просят писать?",
    "money": "Какая из строк — сумма к возврату?",
    "date": "Какая из строк — дата, когда привезут заказ?",
}


def spans(text, kind):
    found = []
    for match in PATTERNS[kind].finditer(text):
        span = match.group(0).strip()
        if span not in found:
            found.append(span)
    return found


def normalize(kind, span):
    if kind == "phone":
        digits = re.sub(r"\D", "", span)
        if digits.startswith("8"):
            return "7" + digits[1:]
        return digits
    if kind == "money":
        return str(int(re.sub(r"\D", "", span)))
    return re.sub(r"\s+", " ", span).strip()


def request_for(text, kind, found):
    criteria = {"none": "ни одна строка не отвечает на вопрос"}
    for i, span in enumerate(found):
        criteria[f"c{i}"] = span
    return {
        "state": text,
        "questions": {"pick": {"type": "choice", "instructions": ASK[kind], "criteria": criteria}},
    }


def extract(judge, text, kind):
    found = spans(text, kind)
    if not found:
        return {"value": "none", "confidence": None}
    answer = judge(request_for(text, kind, found))["answers"]["pick"]
    if answer["choice"] == "none":
        return {"value": "none", "confidence": answer["confidence"]}
    index = int(answer["choice"][1:])
    return {"value": normalize(kind, found[index]), "confidence": answer["confidence"]}


def run(judge):
    hits = 0
    for row in DATA:
        result = extract(judge, row["text"], row["kind"])
        if result["value"] == row["value"]:
            mark = "ok"
            hits += 1
        else:
            mark = "miss"
        if result["confidence"] is None:
            conf = "n/a"
        else:
            conf = f"{result['confidence']:.2f}"
        print(f"{mark:4} got {result['value']:16} gold {row['value']:16} p={conf}  {row['kind']}")
    print(f"\n{hits}/{len(DATA)} normalised values match")


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
