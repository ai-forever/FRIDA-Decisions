"""Turn yes/no and score answers into features for a small logistic regression.

    python examples/text_features/text_features.py
    python examples/text_features/text_features.py --device cpu
    python examples/text_features/text_features.py --onnx
    python examples/text_features/text_features.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as error:
    raise SystemExit("pip install scikit-learn") from error

DATA = [
    {"split": "train", "escalate": 1, "text": "Заказ №4412 пришёл разбитым, верните деньги сегодня, я зол."},
    {"split": "train", "escalate": 1, "text": "Требую возврат за наушники, которые хрипят со второго дня, заказ 7781."},
    {"split": "train", "escalate": 1, "text": "Курьер 12 марта не привёз посылку и трубку не брал. Хочу, чтобы это исправили."},
    {"split": "train", "escalate": 1, "text": "Списали дважды за заказ 1904. Немедленно верните лишнее, это уже не первый раз."},
    {"split": "train", "escalate": 0, "text": "Чайник красивый, кипятит быстро, спасибо за упаковку."},
    {"split": "train", "escalate": 0, "text": "Книга как в описании, доставка вовремя."},
    {"split": "train", "escalate": 0, "text": "Нормальные наушники за эти деньги, пользуюсь неделю."},
    {"split": "train", "escalate": 0, "text": "Спасибо, оператор всё объяснил и заказ нашёлся."},
    {"split": "test", "escalate": 1, "text": "Заказ 3308 протекает, верните оплату, я не буду это чинить сам."},
    {"split": "test", "escalate": 1, "text": "Обещали доставку 4 апреля и не привезли. Исправьте или верните деньги."},
    {"split": "test", "escalate": 0, "text": "Лампа яркая, мужу понравилась, претензий нет."},
    {"split": "test", "escalate": 0, "text": "Обычный кабель, работает, как и ждал."},
    {"split": "test", "escalate": 1, "text": "Пылесос дымит с первого включения, заказ 6620, жду замену и извинение."},
    {"split": "test", "escalate": 0, "text": "Понравился сервис, ответили быстро и по делу."},
    {"split": "train", "escalate": 1, "text": "Хватит кормить отписками: заказ 5150 бракованный, оформите возврат."},
    {"split": "test", "escalate": 0, "text": "Чашка целая, цвет как на фото."},
]
FEATURES = ("angry", "demand", "concrete", "severity")
SEVERITY = [
    "Спокойный отзыв, делать ничего не нужно.",
    "Лёгкое недовольство, можно ответить шаблоном.",
    "Есть претензия, стоит прочитать.",
    "Явная претензия, лучше ответить человеку.",
    "Жёсткая претензия, откладывать нельзя.",
]


def request_for(text):
    return {
        "state": text,
        "questions": {
            "angry": {"type": "noul", "instructions": "Автор раздражён?"},
            "demand": {"type": "noul", "instructions": "Автор требует вернуть деньги или что-то исправить?"},
            "concrete": {"type": "noul", "instructions": "Назван конкретный заказ, товар или дата?"},
            "severity": {"type": "score", "instructions": "Насколько отзыв требует реакции?",
                         "criteria": SEVERITY},
        },
    }


def featurize(answers):
    return [
        answers["angry"]["noul"],
        answers["demand"]["noul"],
        answers["concrete"]["noul"],
        answers["severity"]["score"] / 4.0,
    ]


def majority_label(labels):
    best = labels[0]
    best_count = 0
    for label in set(labels):
        count = labels.count(label)
        if count > best_count:
            best = label
            best_count = count
    return best


def run(judge):
    responses = judge.judge_batch([request_for(row["text"]) for row in DATA])
    train_x = []
    train_y = []
    test_rows = []
    test_x = []
    for row, response in zip(DATA, responses):
        features = featurize(response["answers"])
        if row["split"] == "train":
            train_x.append(features)
            train_y.append(row["escalate"])
        else:
            test_rows.append(row)
            test_x.append(features)
    model = LogisticRegression(max_iter=400)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    hits = 0
    for row, pred in zip(test_rows, predicted):
        if int(pred) == row["escalate"]:
            hits += 1
    majority = majority_label(train_y)
    baseline = 0
    for row in test_rows:
        if row["escalate"] == majority:
            baseline += 1
    coefs = []
    for name, coef in zip(FEATURES, model.coef_[0]):
        coefs.append(f"{name}={coef:.2f}")
    print("features:", ", ".join(FEATURES))
    print("coefficients:", " ".join(coefs))
    print(f"test accuracy {hits}/{len(test_rows)}")
    print(f"majority baseline (label {majority}) {baseline}/{len(test_rows)}")
    for row, pred in zip(test_rows, predicted):
        if int(pred) == row["escalate"]:
            mark = "ok"
        else:
            mark = "miss"
        print(f"  {mark:4} pred {int(pred)} gold {row['escalate']}  {row['text'][:64]}")


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
