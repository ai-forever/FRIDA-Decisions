"""Quickstart: one request with all four question types, then a 243-intent catalog.

    python examples/quickstart.py                       # torch, GPU if available
    python examples/quickstart.py --device cpu
    python examples/quickstart.py --onnx                # int8 ONNX on CPU
    python examples/quickstart.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

TICKET = {
    "state": "Добрый день. Третий день не могу войти в личный кабинет: пишет, что пароль "
             "неверный, а письмо для сброса не приходит. Из-за этого не могу скачать счёт "
             "для бухгалтерии, срок оплаты завтра.",
    "questions": {
        # choice: pick one of several described options
        "team": {"type": "choice", "instructions": "В какую команду направить обращение?",
                 "criteria": {"auth": "доступ к аккаунту, вход, пароли",
                              "billing": "счета, оплата, возвраты",
                              "shipping": "доставка заказов",
                              "other": "всё остальное"}},
        # score: an ordered scale, answered with an expected level
        "urgency": {"type": "score", "instructions": "Насколько срочно нужно ответить?",
                    "criteria": ["не срочно", "в течение недели", "сегодня", "немедленно"]},
        # yes/no: answered with a probability of "yes"
        "angry": {"type": "noul", "instructions": "Автор раздражён?"},
        # ranking: order candidate texts best first
        "next_step": {"type": "ranking", "instructions": "Какой шаг поддержки поможет первым?",
                      "criteria": {"resend": "Повторно отправить письмо для сброса пароля "
                                             "и проверить папку «Спам».",
                                   "invoice": "Выслать счёт на почту вручную.",
                                   "refund": "Оформить возврат средств."}},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default=None, help="cuda or cpu (torch backend)")
    parser.add_argument("--onnx", action="store_true", help="int8 ONNX on CPU")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.onnx:
        from frida_decisions import OnnxJudge
        judge = OnnxJudge.from_pretrained(args.model, threads=args.threads)
    else:
        import torch

        from frida_decisions import Judge
        if args.threads:
            torch.set_num_threads(args.threads)
        judge = Judge.from_pretrained(args.model, device=args.device)

    response = judge(TICKET)
    for qid, answer in response["answers"].items():
        print(qid, json.dumps(answer, ensure_ascii=False))
    print("usage", response["usage"])

    catalog = json.loads((Path(__file__).parent / "data" / "intent_catalog.json")
                         .read_text(encoding="utf-8"))
    counts = judge.token_counts(catalog["request"])
    started = time.perf_counter()
    response = judge(catalog["request"])
    elapsed = time.perf_counter() - started
    answer = response["answers"]["intent"]
    print(f"\n{counts['candidates']} intents -> {answer['choice']} "
          f"(p={answer['probabilities'][answer['choice']]:.2f}, expected {catalog['expected']['intent']}) "
          f"in {elapsed:.2f}s")
    print(f"encoder tokens: naive {counts['naive']}, packed {counts['packed']}, "
          f"with state cache {counts['cached']}; used: {response['usage']}")


if __name__ == "__main__":
    main()
