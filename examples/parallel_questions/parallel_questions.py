"""Ask a dozen questions in one request and compare that with one request each.

    python examples/parallel_questions/parallel_questions.py
    python examples/parallel_questions/parallel_questions.py --device cpu
    python examples/parallel_questions/parallel_questions.py --onnx
    python examples/parallel_questions/parallel_questions.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

import time

POLICY = (
    "Магазин «Северная полка» продаёт книги и посуду. Товар надлежащего качества можно вернуть "
    "в течение 14 дней с даты получения, если сохранены товарный вид, упаковка и чек. "
    "Книги с повреждённой обложкой и посуда со сколом обмену не подлежат: это след использования. "
    "Если товар сломался сам в первые семь дней, магазин меняет его или возвращает деньги, "
    "доставку оплачивает магазин. Заявление пишут на адрес support@severnaya-polka.example, "
    "в теме указывают номер заказа. Деньги возвращают на ту же карту в срок до десяти рабочих дней. "
    "Персональные данные из заявления используются только для возврата и не передаются третьим лицам. "
    "Акции и скидки на этот порядок не влияют: срок 14 дней считается и для товара, купленного со скидкой."
)
QUESTIONS = {
    "kind": {"type": "choice", "instructions": "Что это за текст?",
             "criteria": {"policy": "правила возврата магазина", "ad": "реклама товара",
                          "news": "новость", "manual": "инструкция к прибору"}},
    "audience": {"type": "choice", "instructions": "Для кого написан текст?",
                 "criteria": {"buyer": "покупатель", "courier": "курьер",
                              "accountant": "бухгалтер", "child": "ребёнок"}},
    "contact": {"type": "choice", "instructions": "Как с магазином связаться по этому тексту?",
                "criteria": {"email": "по электронной почте", "phone": "по телефону",
                             "office": "прийти в офис", "chat": "через чат на сайте"}},
    "broken": {"type": "choice", "instructions": "Что магазин делает, если товар сломался сам в первую неделю?",
               "criteria": {"replace": "меняет или возвращает деньги", "repair": "только чинит",
                            "refuse": "отказывает", "credit": "даёт баллы на следующую покупку"}},
    "formality": {"type": "score", "instructions": "Насколько официален тон текста?",
                  "criteria": ["Совсем разговорный.", "Скорее разговорный.", "Нейтральный.",
                               "Скорее официальный.", "Канцелярский."]},
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
FLOAT_TOLERANCE = 0.05


def value_of(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "score":
        return answer["score"]
    return answer["noul"]


def same(left, right):
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= FLOAT_TOLERANCE
    return left == right


def run(judge):
    full = {"state": POLICY, "questions": QUESTIONS}
    print("token_counts", judge.token_counts(full))

    separate = {}
    started = time.perf_counter()
    for key, question in QUESTIONS.items():
        separate[key] = judge({"state": POLICY, "questions": {key: question}})
    elapsed_separate = time.perf_counter() - started

    started = time.perf_counter()
    together = judge(full)
    elapsed_together = time.perf_counter() - started

    widest = 0.0
    mismatches = []
    for key in QUESTIONS:
        left = value_of(together["answers"][key])
        right = value_of(separate[key]["answers"][key])
        if isinstance(left, float) and isinstance(right, float):
            widest = max(widest, abs(left - right))
        if not same(left, right):
            mismatches.append(key)
        print(f"  {key:16} together {left!s:20} apart {right}")

    separate_ms = 0.0
    for item in separate.values():
        separate_ms += item["usage"]["milliseconds"]
    print(f"\n12 requests, cold: {elapsed_separate:.2f}s  sum of milliseconds {separate_ms:.0f}")
    print(f"one request:       {elapsed_together:.2f}s  usage {together['usage']}")
    print("state_truncated", together["usage"]["state_truncated"],
          "state_tokens", together["usage"]["state_tokens"])
    print(f"max abs gap on numeric answers {widest:.4f}")
    if mismatches:
        print("mismatches", mismatches)
    else:
        print(f"all {len(QUESTIONS)} decisions agree")


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
