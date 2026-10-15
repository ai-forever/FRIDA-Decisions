"""Re-rank a BM25 shortlist and drop it when nothing actually answers.

    python examples/rag_filtering/rag_filtering.py
    python examples/rag_filtering/rag_filtering.py --device cpu
    python examples/rag_filtering/rag_filtering.py --onnx
    python examples/rag_filtering/rag_filtering.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
try:
    from rank_bm25 import BM25Okapi
except ImportError as error:
    raise SystemExit("pip install rank_bm25") from error

TOP_K = 6
# Fitted in the README on lists of five with one relevant passage. Re-fit it for
# a list of a different shape; it is not a universal cutoff.
LIST_THRESHOLD = 3.5
NOUL_THRESHOLD = 0.5


def tokens(text: str) -> list[str]:
    return text.lower().replace("ё", "е").split()


def shortlist(question: str, passages: list[dict], k: int) -> list[dict]:
    index = BM25Okapi([tokens(item["text"]) for item in passages])
    scores = index.get_scores(tokens(question))
    order = sorted(range(len(passages)), key=lambda i: -scores[i])[:k]
    return [passages[i] for i in order]


def request_for(question: str, picked: list[dict]) -> dict:
    # The ranking id is not shown to the model. The noul instruction repeats a
    # short passage, so the passages in data.json are one sentence each.
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


def hit(order: list[str], relevant: set[str], k: int):
    if not relevant:
        return None
    return int(any(item in relevant for item in order[:k]))

# %% [markdown]
# ## Relative rank, then an absolute yes/no
#
# Top-k hit rate compares BM25 with `ranking` on queries that have a relevant passage.
# The list threshold only looks at margins inside one shortlist. The noul gate is what
# drops a query whose shortlist does not answer it.

# %%
def run(judge, rows: dict) -> None:
    passages = rows["passages"]
    by_id = {item["id"]: item["text"] for item in passages}
    top1 = {"bm25": [], "ranking": []}
    top3 = {"bm25": [], "ranking": []}
    kept_relevant = kept_total = gold_relevant = 0
    empty_queries = empty_abstained = 0

    for query in rows["queries"]:
        relevant = set(query["relevant"])
        picked = shortlist(query["question"], passages, TOP_K)
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
            top1["bm25"].append(hit(bm25_order, relevant, 1))
            top1["ranking"].append(hit(ranking_order, relevant, 1))
            top3["bm25"].append(hit(bm25_order, relevant, 3))
            top3["ranking"].append(hit(ranking_order, relevant, 3))
            for key in ranking_order:
                if scores[key] - level < LIST_THRESHOLD:
                    continue
                kept_total += 1
                kept_relevant += key in relevant
            gold_relevant += len(relevant & set(bm25_order))
        else:
            empty_queries += 1
            if all(response["answers"][key]["noul"] < NOUL_THRESHOLD for key in ranking_order):
                empty_abstained += 1

        if response["usage"]["state_truncated"]:
            print("  state truncated")

    def mean(values: list) -> str:
        return "n/a" if not values else f"{sum(values) / len(values):.0%}"

    print("\nqueries that have a relevant passage:")
    print(f"  top-1  bm25 {mean(top1['bm25'])}  ranking {mean(top1['ranking'])}")
    print(f"  top-3  bm25 {mean(top3['bm25'])}  ranking {mean(top3['ranking'])}")
    precision = "n/a" if not kept_total else f"{kept_relevant / kept_total:.0%}"
    recall = "n/a" if not gold_relevant else f"{kept_relevant / gold_relevant:.0%}"
    print(f"  list threshold {LIST_THRESHOLD}: kept {kept_total}, "
          f"precision {precision}, recall against the shortlist {recall}")
    print(f"queries with nothing relevant: abstained on noul "
          f"{empty_abstained}/{empty_queries} (cutoff {NOUL_THRESHOLD})")
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
