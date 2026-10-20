"""Score a letter on separate scales and combine the scores with weights in code.

    python examples/composite_scoring/composite_scoring.py
    python examples/composite_scoring/composite_scoring.py --device cpu
    python examples/composite_scoring/composite_scoring.py --onnx
    python examples/composite_scoring/composite_scoring.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

DATA = [
    {"name": "Ира", "invite": True, "text": "Три года писала бэкенд на Django и PostgreSQL. В прошлом квартале сама разобрала инцидент с блокировками в базе: время ответа вернулось с 800 мс к 90 мс. Могу показать разбор."},
    {"name": "Олег", "invite": True, "text": "Пять лет на Python. Django, PostgreSQL, очереди. Последний инцидент: реплика отставала на час, нашёл тяжёлый запрос и вынес его из транзакции. Пишу коротко, детали готов обсудить."},
    {"name": "Маша", "invite": False, "text": "Я повар. Ищу работу на кухне ресторана. Python, Django и базы данных мне не нужны, я ими не занималась."},
    {"name": "Кирилл", "invite": False, "text": "Делал вёрстку на React. С Python знаком по учебному курсу, базу данных в проде не сопровождал. Хочу перейти в бэкенд."},
    {"name": "Нина", "invite": False, "text": "Django видела, PostgreSQL ставила локально. Инциденты не разбирала, в проде не дежурила. Письмо длинное, потому что хочу рассказать про хобби и путешествия, это важно для культуры."},
    {"name": "Павел", "invite": True, "text": "Четыре года поддерживал Django-монолит и PostgreSQL на 2 ТБ. Ночные инциденты: дважды откатывал миграцию, один раз чинил дедлок по логам. Без воды: что сломалось, что сделал, чем кончилось."},
]
ROLE = "Вакансия: бэкенд на Python, нужны Django, PostgreSQL и опыт разбора чужих инцидентов."
RUBRICS = {
    "skills": [
        "Опыта по делу нет.",
        "Смежный опыт, нужного стека почти нет.",
        "Часть стека есть, без конкретных задач.",
        "Нужный стек и названы конкретные задачи.",
        "Стек, цифры и самостоятельные решения.",
    ],
    "clarity": [
        "Непонятно, что человек умеет.",
        "Общие слова, почти без фактов.",
        "Факты есть, но их приходится вычитывать.",
        "Письмо по делу и его легко дочитать.",
        "Коротко, по делу, с примерами.",
    ],
    "tone": [
        "Грубо или пренебрежительно.",
        "Фамильярно, рабочему письму не подходит.",
        "Нейтрально, без явных сбоев.",
        "Спокойно и уважительно.",
        "Уважительно и без заискивания.",
    ],
}
WEIGHTS = {
    "stack": {"skills": 0.7, "clarity": 0.2, "tone": 0.1},
    "writing": {"skills": 0.2, "clarity": 0.5, "tone": 0.3},
}
CUTOFF = 1.15


def request_for(text):
    return {
        "state": ROLE + "\n\nПисьмо кандидата:\n" + text,
        "questions": {
            "skills": {"type": "score", "instructions": "Насколько опыт совпадает с вакансией?",
                       "criteria": RUBRICS["skills"]},
            "clarity": {"type": "score", "instructions": "Насколько письмо ясное и конкретное?",
                        "criteria": RUBRICS["clarity"]},
            "tone": {"type": "score", "instructions": "Насколько тон уместен для рабочего письма?",
                     "criteria": RUBRICS["tone"]},
        },
    }


def weighted(answers, weights):
    total = 0.0
    for name in weights:
        total += weights[name] * answers[name]["score"]
    return total


def run(judge):
    scored = []
    for row in DATA:
        answers = judge(request_for(row["text"]))["answers"]
        values = {}
        for name, weights in WEIGHTS.items():
            values[name] = weighted(answers, weights)
        scored.append({"name": row["name"], "invite": row["invite"], "answers": answers, "values": values})

    print(f"{'name':10} {'skills':6} {'clarity':7} {'tone':6} {'stack':7} {'writing':7} invite")
    hits = 0
    for row in scored:
        answers = row["answers"]
        invite = row["values"]["stack"] >= CUTOFF
        if invite == row["invite"]:
            hits += 1
        print(f"{row['name']:10} {answers['skills']['score']:.2f}   {answers['clarity']['score']:.2f}    "
              f"{answers['tone']['score']:.2f}   {row['values']['stack']:.2f}    "
              f"{row['values']['writing']:.2f}    {str(invite):5} gold {row['invite']}")

    by_stack = sorted(scored, key=lambda row: -row["values"]["stack"])
    by_writing = sorted(scored, key=lambda row: -row["values"]["writing"])
    print("\norder by stack weights:   " + ", ".join(row["name"] for row in by_stack))
    print("order by writing weights: " + ", ".join(row["name"] for row in by_writing))
    print(f"cutoff {CUTOFF} on the stack weights agrees with the invite label in {hits}/{len(DATA)}")


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
