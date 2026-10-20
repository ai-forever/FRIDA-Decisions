"""Re-rank a BM25 shortlist and drop it when nothing actually answers.

    python examples/rag_filtering/rag_filtering.py
    python examples/rag_filtering/rag_filtering.py --device cpu
    python examples/rag_filtering/rag_filtering.py --onnx
    python examples/rag_filtering/rag_filtering.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

try:
    from rank_bm25 import BM25Okapi
except ImportError as error:
    raise SystemExit("pip install rank_bm25") from error

PASSAGES = [
    {"id": "k1", "text": "Налейте в чайник воду с двумя ложками лимонной кислоты, вскипятите и оставьте на час, затем промойте."},
    {"id": "k2", "text": "Уксус растворяет накипь: смешайте его с водой один к одному, прокипятите чайник и сполосните."},
    {"id": "k3", "text": "Накипь образуется из солей кальция и магния, растворённых в жёсткой воде."},
    {"id": "k4", "text": "Возврат денег за двойное списание оформляют в личном кабинете, в разделе платежей."},
    {"id": "k5", "text": "Чтобы вернуть деньги за заказ, откройте заказ и нажмите «Оформить возврат», срок три дня."},
    {"id": "k6", "text": "Чек об оплате приходит на почту сразу после списания средств."},
    {"id": "k7", "text": "Письмо для сброса пароля приходит в течение пяти минут, проверьте папку «Спам»."},
    {"id": "k8", "text": "Сменить пароль можно в профиле, если вы ещё вошли в аккаунт на другом устройстве."},
    {"id": "k9", "text": "Доставка по городу занимает от одного до трёх рабочих дней, курьер звонит за час."},
    {"id": "k10", "text": "Трек-номер появляется в заказе, когда магазин передаёт посылку в службу доставки."},
    {"id": "k11", "text": "Фильтр-кувшин смягчает воду, и бытовые приборы реже покрываются накипью."},
    {"id": "k12", "text": "Гарантия не покрывает сколы от удара и следы неправильной эксплуатации."},
]
QUERIES = [
    {"question": "Как удалить накипь из чайника?", "relevant": ["k1", "k2"]},
    {"question": "Как вернуть деньги за заказ?", "relevant": ["k4", "k5"]},
    {"question": "Не приходит письмо для сброса пароля.", "relevant": ["k7"]},
    {"question": "Сколько ждать доставку по городу?", "relevant": ["k9"]},
    {"question": "Какая столица Франции?", "relevant": []},
    {"question": "Как сварить пасту аль денте?", "relevant": []},
]
TOP_K = 6
LIST_THRESHOLD = 3.5
NOUL_THRESHOLD = 0.5


def tokens(text):
    return text.lower().replace("ё", "е").split()


def shortlist(question, k):
    index = BM25Okapi([tokens(item["text"]) for item in PASSAGES])
    scores = index.get_scores(tokens(question))
    order = sorted(range(len(PASSAGES)), key=lambda i: -float(scores[i]))
    picked = []
    for i in order[:k]:
        picked.append(PASSAGES[i])
    return picked


def request_for(question, picked):
    questions = {
        "best": {
            "type": "ranking",
            "instructions": "Какой фрагмент лучше всего отвечает на вопрос?",
            "criteria": {item["id"]: item["text"] for item in picked},
        },
    }
    for item in picked:
        questions[item["id"]] = {
            "type": "noul",
            "instructions": "Этот фрагмент отвечает на вопрос? " + item["text"],
        }
    return {"state": question, "questions": questions}


def hit(order, relevant, k):
    if not relevant:
        return None
    for item in order[:k]:
        if item in relevant:
            return 1
    return 0


def rate(values):
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.0%}"


def report_metrics(stats):
    top1, top3 = stats["top1"], stats["top3"]
    kept_relevant = stats["kept_relevant"]
    kept_total = stats["kept_total"]
    print("\nqueries that have a relevant passage:")
    print(f"  top-1  bm25 {rate(top1['bm25'])}  ranking {rate(top1['ranking'])}")
    print(f"  top-3  bm25 {rate(top3['bm25'])}  ranking {rate(top3['ranking'])}")
    if kept_total:
        precision = f"{kept_relevant / kept_total:.0%}"
    else:
        precision = "n/a"
    if stats["gold_relevant"]:
        recall = f"{kept_relevant / stats['gold_relevant']:.0%}"
    else:
        recall = "n/a"
    print(f"  list threshold {LIST_THRESHOLD}: kept {kept_total}, "
          f"precision {precision}, recall against the shortlist {recall}")
    print(f"queries with nothing relevant: abstained on noul "
          f"{stats['empty_abstained']}/{stats['empty_queries']} (cutoff {NOUL_THRESHOLD})")


def run(judge):
    stats = {
        "top1": {"bm25": [], "ranking": []},
        "top3": {"bm25": [], "ranking": []},
        "kept_relevant": 0,
        "kept_total": 0,
        "gold_relevant": 0,
        "empty_queries": 0,
        "empty_abstained": 0,
    }

    for query in QUERIES:
        relevant = set(query["relevant"])
        picked = shortlist(query["question"], TOP_K)
        bm25_order = [item["id"] for item in picked]
        response = judge(request_for(query["question"], picked))
        answer = response["answers"]["best"]
        scores = answer["scores"]
        level = sum(scores.values()) / len(scores)
        ranking_order = answer["ranking"]
        print(f"\n{query['question']}")
        print(f"  bm25    {bm25_order}")
        for key in ranking_order:
            distance = scores[key] - level
            noul = response["answers"][key]["noul"]
            flags = []
            if distance >= LIST_THRESHOLD:
                flags.append("above-list")
            if noul >= NOUL_THRESHOLD:
                flags.append("noul")
            if key in relevant:
                flags.append("gold")
            print(f"  {key:4} margin {scores[key]:+6.2f}  above list {distance:+5.2f}  "
                  f"noul {noul:.2f}  {' '.join(flags)}")
        if relevant:
            stats["top1"]["bm25"].append(hit(bm25_order, relevant, 1))
            stats["top1"]["ranking"].append(hit(ranking_order, relevant, 1))
            stats["top3"]["bm25"].append(hit(bm25_order, relevant, 3))
            stats["top3"]["ranking"].append(hit(ranking_order, relevant, 3))
            for key in ranking_order:
                if scores[key] - level < LIST_THRESHOLD:
                    continue
                stats["kept_total"] += 1
                if key in relevant:
                    stats["kept_relevant"] += 1
            stats["gold_relevant"] += len(relevant & set(bm25_order))
        else:
            stats["empty_queries"] += 1
            answered = False
            for key in ranking_order:
                if response["answers"][key]["noul"] >= NOUL_THRESHOLD:
                    answered = True
            if not answered:
                stats["empty_abstained"] += 1

    report_metrics(stats)


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
