"""Let a regex propose spans and a choice question pick the one that was asked for.

    python examples/value_extraction/value_extraction.py
    python examples/value_extraction/value_extraction.py --device cpu
    python examples/value_extraction/value_extraction.py --onnx
    python examples/value_extraction/value_extraction.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
import re

PATTERNS = {
    "phone": re.compile(r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
    "money": re.compile(r"\d[\d\s]*\d\s*(?:руб(?:лей|ля|\.)?|₽)"),
    "date": re.compile(r"\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
                       r"сентября|октября|ноября|декабря)(?:\s+\d{4})?"),
}
ASK = {
    "phone": "Какая из строк — телефон, на который просят писать?",
    "money": "Какая из строк — сумма к возврату?",
    "date": "Какая из строк — дата, когда привезут заказ?",
}


def spans(text: str, kind: str) -> list[str]:
    return list(dict.fromkeys(match.group(0).strip() for match in PATTERNS[kind].finditer(text)))


def normalize(kind: str, span: str) -> str:
    if kind == "phone":
        digits = re.sub(r"\D", "", span)
        if digits.startswith("8"):
            digits = "7" + digits[1:]
        return digits
    if kind == "money":
        return str(int(re.sub(r"\D", "", span)))
    return re.sub(r"\s+", " ", span).strip()


def request_for(text: str, kind: str, found: list[str]) -> dict:
    criteria = {f"c{i}": span for i, span in enumerate(found)}
    criteria["none"] = "ни одна строка не отвечает на вопрос"
    return {
        "state": text,
        "questions": {"pick": {"type": "choice", "instructions": ASK[kind], "criteria": criteria}},
    }


def extract(judge, text: str, kind: str) -> dict:
    found = spans(text, kind)
    if not found:
        return {"value": "none", "choice": "none", "confidence": None}
    response = judge(request_for(text, kind, found))
    answer = response["answers"]["pick"]
    choice = answer["choice"]
    if choice == "none":
        return {"value": "none", "choice": choice, "confidence": answer["confidence"]}
    index = int(choice[1:])
    return {"value": normalize(kind, found[index]), "choice": choice,
            "confidence": answer["confidence"], "span": found[index]}

# %% [markdown]
# ## The model points, the regex owns the characters
#
# `value` is the normalised form of a span that occurs in the text, or `none`.

# %%
def run(judge, rows: list[dict]) -> None:
    hits = 0
    for row in rows:
        result = extract(judge, row["text"], row["kind"])
        hits += result["value"] == row["value"]
        conf = "n/a" if result["confidence"] is None else f"{result['confidence']:.2f}"
        print(f"{'ok' if result['value'] == row['value'] else 'miss':4} "
              f"got {result['value']:16} gold {row['value']:16} p={conf}  {row['kind']}")
    print(f"\n{hits}/{len(rows)} normalised values match")
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
