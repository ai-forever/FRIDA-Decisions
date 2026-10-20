"""Route a support ticket only when the team choice is confident enough.

    python examples/support_triage/support_triage.py
    python examples/support_triage/support_triage.py --device cpu
    python examples/support_triage/support_triage.py --onnx
    python examples/support_triage/support_triage.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

DATA = [
    {"team": "auth", "text": "Не могу войти в личный кабинет: пишет, что пароль неверный, а письмо для сброса не приходит."},
    {"team": "billing", "text": "Вчера оплатил заказ картой, деньги списались дважды. Верните, пожалуйста, лишнее списание."},
    {"team": "shipping", "text": "Курьер должен был привезти заказ вчера, трек-номер не обновляется второй день."},
    {"team": "product", "text": "Чайник пришёл с трещиной на колбе, кипятить его нельзя. Хочу обмен на такой же."},
    {"team": "billing", "text": "Подписка продлилась сама, я её не продлял. Снимите списание за этот месяц."},
    {"team": "auth", "text": "Поменял номер телефона и теперь не могу подтвердить вход: код приходит на старый номер."},
    {"team": "product", "text": "В коробке нет зарядки, которая указана в комплекте. Сама доставка пришла вовремя, не хватает вещи."},
    {"team": "product", "text": "Наушники хрипят на левом канале с первого дня. В описании этого не было."},
    {"team": "billing", "text": "Чек на сайте показывает одну сумму, а в банке списалось больше. Пришлите правильный чек."},
    {"team": "shipping", "text": "Нужно изменить адрес доставки: курьер ещё не выехал, заказ только собирают на складе."},
    {"team": "auth", "text": "После смены почты не приходит ни код входа, ни уведомления. Доступа к старому ящику больше нет."},
    {"team": "shipping", "text": "Курьер опоздал на три часа и даже не позвонил. Больше заказывать у вас не буду."},
]

TEAMS = {
    "auth": "доступ к аккаунту, вход, пароль, личные данные",
    "billing": "оплата, списания, возвраты денег, подписки",
    "shipping": "доставка, курьер, сроки и адрес получения заказа",
    "product": "качество, комплектность и характеристики товара",
}
URGENCY = [
    "Очень низкая: работа не страдает, ответить можно когда угодно.",
    "Низкая: неудобство небольшое, ответить можно на этой неделе.",
    "Средняя: работа затруднена, но обходной путь есть.",
    "Высокая: работа заблокирована, обходного пути нет.",
    "Критическая: потери растут с каждым часом.",
]
DEFAULT_CONFIDENCE = 0.7
HIGH_URGENCY = 3.0
UPSET = 0.6


def request_for(text):
    return {
        "state": text,
        "questions": {
            "team": {"type": "choice", "instructions": "В какую команду направить обращение?",
                     "criteria": TEAMS},
            "urgency": {"type": "score", "instructions": "Оцени срочность обращения по шкале ниже.",
                        "criteria": URGENCY},
            "angry": {"type": "noul", "instructions": "Автор раздражён?"},
            "churn": {"type": "noul", "instructions": "Автор грозит уйти или больше не покупать?"},
        },
    }


def route(answers, min_confidence):
    team = answers["team"]
    urgent = answers["urgency"]["score"] >= HIGH_URGENCY
    upset = answers["angry"]["noul"] >= UPSET or answers["churn"]["noul"] >= UPSET
    auto = team["confidence"] >= min_confidence
    decision = {"auto": auto, "confidence": team["confidence"]}
    if auto:
        decision["queue"] = team["choice"]
    else:
        decision["queue"] = "operator"
    if urgent or upset:
        decision["priority"] = "high"
    else:
        decision["priority"] = "normal"
    return decision


def coverage(scored, min_confidence):
    """Accuracy is only over tickets the model routes itself."""
    auto = 0
    correct = 0
    for row in scored:
        decision = route(row["answers"], min_confidence)
        if not decision["auto"]:
            continue
        auto += 1
        if decision["queue"] == row["team"]:
            correct += 1
    return auto, correct


def run(judge):
    responses = judge.judge_batch([request_for(row["text"]) for row in DATA])
    scored = []
    print(f"{'result':8} {'queue':10} {'pri':6} conf  urgency angry churn  gold")
    for row, response in zip(DATA, responses):
        answers = response["answers"]
        decision = route(answers, DEFAULT_CONFIDENCE)
        scored.append({"team": row["team"], "answers": answers})
        if not decision["auto"]:
            mark = "hold"
        elif decision["queue"] == row["team"]:
            mark = "ok"
        else:
            mark = "miss"
        print(f"{mark:8} {decision['queue']:10} {decision['priority']:6} "
              f"{decision['confidence']:.2f}  {answers['urgency']['score']:.2f}    "
              f"{answers['angry']['noul']:.2f}  {answers['churn']['noul']:.2f}   {row['team']}")

    print(f"\nthreshold {DEFAULT_CONFIDENCE}:")
    auto, correct = coverage(scored, DEFAULT_CONFIDENCE)
    if auto:
        accuracy = f"{correct}/{auto}"
    else:
        accuracy = "n/a"
    print(f"  auto {auto}/{len(scored)} ({auto / len(scored):.0%}), accuracy on auto {accuracy}")
    print("\nthreshold sweep:")
    for threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        auto, correct = coverage(scored, threshold)
        if auto:
            accuracy = f"{correct / auto:.0%}"
        else:
            accuracy = "n/a"
        print(f"  {threshold:.2f}  coverage {auto / len(scored):.0%}  accuracy {accuracy}")


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
