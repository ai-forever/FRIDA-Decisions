"""Ask a dozen questions in one request and compare that with one request each.

    python examples/parallel_questions/parallel_questions.py
    python examples/parallel_questions/parallel_questions.py --device cpu
    python examples/parallel_questions/parallel_questions.py --onnx
    python examples/parallel_questions/parallel_questions.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
import time

QUESTIONS = {
    "kind": {"type": "choice", "instructions": "Что это за текст?",
             "criteria": {"policy": "правила возврата магазина",
                          "ad": "реклама товара",
                          "news": "новость",
                          "manual": "инструкция к прибору"}},
    "audience": {"type": "choice", "instructions": "Для кого написан текст?",
                 "criteria": {"buyer": "покупатель", "courier": "курьер",
                              "accountant": "бухгалтер", "child": "ребёнок"}},
    "contact": {"type": "choice", "instructions": "Как с магазином связаться по этому тексту?",
                "criteria": {"email": "по электронной почте", "phone": "по телефону",
                             "office": "прийти в офис", "chat": "через чат на сайте"}},
    "broken": {"type": "choice", "instructions": "Что магазин делает, если товар сломался сам в первую неделю?",
               "criteria": {"replace": "меняет или возвращает деньги",
                            "repair": "только чинит",
                            "refuse": "отказывает",
                            "credit": "даёт баллы на следующую покупку"}},
    "formality": {"type": "score", "instructions": "Насколько официален тон текста?",
                  "criteria": ["Совсем разговорный.", "Скорее разговорный.",
                               "Нейтральный.", "Скорее официальный.", "Канцелярский."]},
    "strictness": {"type": "score", "instructions": "Насколько жёсткие условия возврата?",
                   "criteria": ["Вернуть можно почти всё и почти всегда.",
                                "Условия мягкие, исключений мало.",
                                "Есть и срок, и заметные исключения.",
                                "Условия жёсткие, исключений много.",
                                "Вернуть практически нельзя."]},
    "refund": {"type": "noul", "instructions": "В тексте есть возврат денег или товара?"},
    "days14": {"type": "noul", "instructions": "Назван срок в 14 дней?"},
    "personal": {"type": "noul", "instructions": "Текст говорит, как поступят с персональными данными?"},
    "store_pays": {"type": "noul", "instructions": "Доставку при обмене брака оплачивает магазин?"},
    "sells_books": {"type": "noul", "instructions": "Магазин продаёт книги?"},
    "discount_blocks": {"type": "noul", "instructions": "Скидка отменяет право на возврат?"},
}


def decision(answer: dict):
    kind = answer["type"]
    if kind == "choice":
        return answer["choice"]
    if kind == "score":
        return round(answer["score"], 4)
    if kind == "noul":
        return round(answer["noul"], 4)
    return tuple(answer.get("ranking", ()))


def gap(left, right):
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right)
    return 0.0 if left == right else None

# %% [markdown]
# ## Separate calls first, then the same questions together
#
# The separate calls run first, on a cold state cache, so a packed call cannot warm the cache for them. `token_counts` is the structural comparison: `naive` re-encodes the text once per option, `packed` keeps one copy per row. Wall clock on a short text is a smaller effect than the token counts. Floats from the int8 ONNX graph can move in the second decimal between the two shapes; a choice has to match exactly.

# %%
def run(judge, rows: dict) -> None:
    state = rows["state"]
    full = {"state": state, "questions": QUESTIONS}
    counts = judge.token_counts(full)
    print("token_counts", counts)

    separate = {}
    started = time.perf_counter()
    for key, question in QUESTIONS.items():
        separate[key] = judge({"state": state, "questions": {key: question}})
    elapsed_separate = time.perf_counter() - started

    started = time.perf_counter()
    together = judge(full)
    elapsed_together = time.perf_counter() - started

    widest = 0.0
    mismatches = []
    for key in QUESTIONS:
        left = decision(together["answers"][key])
        right = decision(separate[key]["answers"][key])
        delta = gap(left, right)
        if delta is None:
            mismatches.append(key)
        else:
            widest = max(widest, delta)
        print(f"  {key:16} together {left!s:20} apart {right}")

    print(f"\n12 requests, cold: {elapsed_separate:.2f}s  "
          f"sum of milliseconds {sum(item['usage']['milliseconds'] for item in separate.values()):.0f}")
    print(f"one request:       {elapsed_together:.2f}s  usage {together['usage']}")
    print("state_truncated", together["usage"]["state_truncated"],
          "state_tokens", together["usage"]["state_tokens"])
    print(f"max abs gap on numeric answers {widest:.4f}")
    if mismatches:
        print("choice mismatches", mismatches)
    else:
        print(f"all {len(QUESTIONS)} decisions agree")
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
