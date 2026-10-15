"""Walk a product tree with a beam over choice probabilities.

    python examples/hierarchical_classification/hierarchical_classification.py
    python examples/hierarchical_classification/hierarchical_classification.py --device cpu
    python examples/hierarchical_classification/hierarchical_classification.py --onnx
    python examples/hierarchical_classification/hierarchical_classification.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
TREE = {
    "goods": {
        "electronics": "электроника и гаджеты",
        "home": "товары для дома и кухни",
        "grocery": "продукты и напитки",
    },
    "electronics": {
        "phones": "телефоны",
        "audio": "наушники и колонки",
    },
    "home": {
        "kitchen": "техника для кухни",
        "furniture": "мебель",
    },
    "grocery": {
        "tea": "чай и кофе",
        "dairy": "молочные продукты",
    },
    "phones": {
        "smartphones": "смартфоны с сенсорным экраном",
        "button": "кнопочные телефоны",
    },
    "audio": {
        "headphones": "наушники",
        "speakers": "колонки",
    },
    "kitchen": {
        "kettles": "чайники и термопоты",
        "coffee": "кофеварки и кофемолки",
    },
    "furniture": {
        "chairs": "стулья",
        "tables": "столы",
    },
    "tea": {
        "black": "чёрный чай",
        "green": "зелёный чай",
    },
    "dairy": {
        "milk": "молоко",
        "cheese": "сыр",
    },
}

BEAM = 2
MIN_CONFIDENCE = 0.45


def classify(judge, text: str) -> dict:
    """Beam over choice probabilities. Low confidence retreats to the parent."""
    frontier = [("goods", 1.0)]
    parent = {"goods": None}
    step_confidence: dict[str, float] = {}
    step_chosen: dict[str, bool] = {}
    while True:
        expandable = [(node, prob) for node, prob in frontier if node in TREE]
        if not expandable:
            break
        questions = {
            node: {"type": "choice", "instructions": "К какой категории из списка относится товар?",
                   "criteria": TREE[node]}
            for node, _ in expandable
        }
        answers = judge({"state": text, "questions": questions})["answers"]
        nxt = []
        for node, prob in expandable:
            answer = answers[node]
            for child, child_p in answer["probabilities"].items():
                parent[child] = node
                step_confidence[child] = answer["confidence"]
                step_chosen[child] = child == answer["choice"]
                nxt.append((child, prob * child_p))
        nxt.sort(key=lambda item: -item[1])
        frontier = nxt[:BEAM]
    best = max(frontier, key=lambda item: item[1])[0]
    label = best
    while label != "goods" and (not step_chosen.get(label, False)
                                or step_confidence.get(label, 0.0) < MIN_CONFIDENCE):
        label = parent[label]
    return {"leaf": best, "label": label, "path_p": max(item[1] for item in frontier),
            "confidence": step_confidence.get(best)}

# %% [markdown]
# ## Exact leaf, or at least an ancestor
#
# `leaf` is the end of the best path before the fallback. `label` is what the code would store. A gold hit counts when the stored label is the gold leaf or one of its ancestors.

# %%
def run(judge, rows: list[dict]) -> None:
    parent = {}
    for node, kids in TREE.items():
        for child in kids:
            parent[child] = node

    def chain(node: str) -> set[str]:
        found = {node}
        while node in parent:
            node = parent[node]
            found.add(node)
        return found

    exact = coarse = 0
    for row in rows:
        result = classify(judge, row["text"])
        exact += result["label"] == row["gold"]
        coarse += result["label"] in chain(row["gold"])
        mark = "exact" if result["label"] == row["gold"] else ("coarse" if result["label"] in chain(row["gold"]) else "miss")
        print(f"{mark:6} gold {row['gold']:12} label {result['label']:12} "
              f"leaf {result['leaf']:12} path_p={result['path_p']:.3f} "
              f"leaf_conf={None if result['confidence'] is None else round(result['confidence'], 3)}")
    print(f"\nexact {exact}/{len(rows)}, gold or an ancestor of gold {coarse}/{len(rows)}, "
          f"beam {BEAM}, confidence cutoff {MIN_CONFIDENCE}")
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
