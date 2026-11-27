"""Quickstart: a first request, the four question types, filtering, a 243-intent catalog.

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
    "state": "Добрый день. Третий день не могу войти в личный кабинет: пишет, что пароль неверный, "
             "а письмо для сброса не приходит. Из-за этого не могу скачать счёт, срок оплаты завтра.",
    "questions": {
        "team": {"type": "choice", "instructions": "В какую команду направить обращение?",
                 "criteria": {"auth": "доступ к аккаунту, вход, пароли",
                              "billing": "счета, оплата, возвраты",
                              "shipping": "доставка заказов"}},
        "angry": {"type": "noul", "instructions": "Автор раздражён?"},
    },
}

# One request per question type: choice, score, noul (yes/no) and ranking.
EXAMPLES = {
    "choice": {
        "state": "Здравствуйте! Вчера оплатил заказ №5512 картой, деньги списались дважды. Верните, пожалуйста, лишнее списание.",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Какая тема обращения?",
                "criteria": {
                    "billing": "оплата, списания, возвраты денег",
                    "delivery": "доставка и сроки получения заказа",
                    "account": "вход в аккаунт, пароль, личные данные",
                    "product": "качество и характеристики товара"
                }
            }
        }
    },
    "score": {
        "state": "Сервер с базой заказов не отвечает уже сорок минут, клиенты не могут оформить покупку, каждая минута — потерянные продажи.",
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "Оцени срочность обращения по шкале ниже.",
                "criteria": [
                    "Очень низкая: работа не страдает, ответить можно когда угодно.",
                    "Низкая: неудобство небольшое, ответить можно на этой неделе.",
                    "Средняя: работа затруднена, но обходной путь есть.",
                    "Высокая: работа заблокирована, обходного пути нет.",
                    "Критическая: потери растут с каждым часом."
                ]
            }
        }
    },
    "noul": {
        "state": "Курьер опоздал на два часа и даже не позвонил. Больше заказывать у вас не буду.",
        "questions": {
            "complaint": {
                "type": "noul",
                "instructions": "Это жалоба?"
            }
        }
    },
    "ranking": {
        "state": "Как удалить накипь из чайника?",
        "questions": {
            "best": {
                "type": "ranking",
                "instructions": "Какой фрагмент лучше всего отвечает на вопрос?",
                "criteria": {
                    "p1": "Налейте в чайник воду с двумя ложками лимонной кислоты, вскипятите и оставьте на час, затем промойте.",
                    "p2": "Электрический чайник нельзя погружать в воду целиком: это опасно для нагревательного элемента.",
                    "p3": "Чай лучше заваривать водой температурой 85–95 градусов, а не крутым кипятком.",
                    "p4": "Уксус тоже растворяет накипь: смешайте его с водой один к одному, прокипятите и тщательно сполосните."
                }
            }
        }
    }
}


# Filtering, not just ordering: a retriever's top-k for the same question, where
# the two passages that answer it are a minority. The four candidates above are
# in here unchanged, which is also the point -- a candidate's margin does not
# depend on what else is in the list.
RETRIEVED = dict(
    EXAMPLES["ranking"]["questions"]["best"]["criteria"],
    p5="Накипь образуется из солей кальция и магния, растворённых в воде; в жёсткой воде её больше.",
    p6="Гарантия на бытовую технику не покрывает поломки из-за нарушения правил эксплуатации.",
    p7="Для заварки зелёного чая возьмите воду около 80 градусов и настаивайте две минуты.",
    p8="Фильтр-кувшин смягчает воду и поэтому реже приходится чистить бытовые приборы.",
    p9="Доставка заказов по городу занимает от одного до трёх рабочих дней.",
)
# How far above its own list a candidate has to stand. Fitted on razvilka, on
# lists where one candidate in five is relevant (`tools/ranking_threshold.py`);
# re-fit it on lists the shape of yours.
THRESHOLD = 3.5


def relevance(judge, criteria: dict) -> list[tuple[str, float, float]]:
    """(id, margin, distance above the list) for one ranking request, best first.

    `scores` are the raw margins, and only differences inside one request are
    meaningful: the level of a request is not calibrated. The level of the
    request's own candidates estimates it, because in a retriever's top-k most
    candidates are irrelevant.
    """
    request = {"state": "Как удалить накипь из чайника?",
               "questions": {"best": {
                   "type": "ranking",
                   "instructions": "Какой фрагмент лучше всего отвечает на вопрос?",
                   "criteria": criteria}}}
    scores = judge(request)["answers"]["best"]["scores"]
    level = sum(scores.values()) / len(scores)
    return sorted(((key, score, score - level) for key, score in scores.items()),
                  key=lambda row: -row[1])


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

    # Several questions about one text, answered together.
    response = judge(TICKET)
    for qid, answer in response["answers"].items():
        print(qid, json.dumps(answer, ensure_ascii=False))
    print("usage", response["usage"])

    # The four question types.
    for kind, request in EXAMPLES.items():
        (qid, answer), = judge(request)["answers"].items()
        print(f"{kind}:", json.dumps(answer, ensure_ascii=False))

    # Filtering passages by a threshold, and what the threshold depends on.
    print(f"\nretrieved {len(RETRIEVED)} passages, keeping those more than "
          f"{THRESHOLD} above their own list:")
    for key, margin, distance in relevance(judge, RETRIEVED):
        kept = "kept" if distance >= THRESHOLD else ""
        print(f"  {key}  margin {margin:+6.2f}  above the list {distance:+6.2f}  {kept}")

    # The same question over the four passages alone. Two of the four answer it,
    # so they pull the level of the list up and the same threshold keeps nobody:
    # it belongs to the shape of the list, not to the model.
    four = relevance(judge, EXAMPLES["ranking"]["questions"]["best"]["criteria"])
    print(f"  the same four passages on their own: best is {four[0][0]} at "
          f"{four[0][2]:+.2f} above the list, so nothing clears {THRESHOLD}")

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
