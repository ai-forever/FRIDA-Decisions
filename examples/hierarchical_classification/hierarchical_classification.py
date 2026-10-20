"""Walk a product tree with a beam over choice probabilities.

    python examples/hierarchical_classification/hierarchical_classification.py
    python examples/hierarchical_classification/hierarchical_classification.py --device cpu
    python examples/hierarchical_classification/hierarchical_classification.py --onnx
    python examples/hierarchical_classification/hierarchical_classification.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

DATA = [
    {"gold": "smartphones", "text": "Смартфон с сенсорным экраном, 128 ГБ, две сим-карты."},
    {"gold": "button", "text": "Кнопочный телефон с крупными кнопками и без сенсорного экрана."},
    {"gold": "headphones", "text": "Беспроводные наушники с шумоподавлением, кейс в комплекте."},
    {"gold": "kettles", "text": "Электрический чайник на 1,7 литра, металлическая колба."},
    {"gold": "chairs", "text": "Офисный стул на колёсиках, высота сиденья регулируется."},
    {"gold": "black", "text": "Пачка чёрного чая, сто пакетиков, без добавок."},
    {"gold": "speakers", "text": "Портативная колонка с блютузом, для музыки в комнате."},
    {"gold": "milk", "text": "Молоко 3,2 процента, один литр, в бутылке."},
]
TREE = {
    "goods": {"electronics": "электроника и гаджеты", "home": "товары для дома и кухни",
              "grocery": "продукты и напитки"},
    "electronics": {"phones": "телефоны", "audio": "наушники и колонки"},
    "home": {"kitchen": "техника для кухни", "furniture": "мебель"},
    "grocery": {"tea": "чай и кофе", "dairy": "молочные продукты"},
    "phones": {"smartphones": "смартфоны с сенсорным экраном", "button": "кнопочные телефоны"},
    "audio": {"headphones": "наушники", "speakers": "колонки"},
    "kitchen": {"kettles": "чайники и термопоты", "coffee": "кофеварки и кофемолки"},
    "furniture": {"chairs": "стулья", "tables": "столы"},
    "tea": {"black": "чёрный чай", "green": "зелёный чай"},
    "dairy": {"milk": "молоко", "cheese": "сыр"},
}
PARENT = {child: node for node, kids in TREE.items() for child in kids}
BEAM = 2
MIN_CONFIDENCE = 0.45


def classify(judge, text):
    frontier = [("goods", 1.0)]
    entered = {}
    while True:
        expandable = []
        for node, prob in frontier:
            if node in TREE:
                expandable.append((node, prob))
        if not expandable:
            break
        questions = {}
        for node, _prob in expandable:
            questions[node] = {
                "type": "choice",
                "instructions": "К какой категории из списка относится товар?",
                "criteria": TREE[node],
            }
        answers = judge({"state": text, "questions": questions})["answers"]
        nxt = []
        for node, prob in expandable:
            answer = answers[node]
            for child, child_p in answer["probabilities"].items():
                if child == answer["choice"]:
                    entered[child] = answer["confidence"]
                nxt.append((child, prob * child_p))
        nxt.sort(key=lambda item: -item[1])
        frontier = nxt[:BEAM]
    best, path_p = frontier[0]
    label = best
    while label != "goods" and entered.get(label, 0.0) < MIN_CONFIDENCE:
        label = PARENT[label]
    return {"leaf": best, "label": label, "path_p": path_p, "confidence": entered.get(best)}


def covers(label, gold):
    node = gold
    while node is not None:
        if node == label:
            return True
        node = PARENT.get(node)
    return False


def run(judge):
    exact = 0
    coarse = 0
    for row in DATA:
        result = classify(judge, row["text"])
        if result["label"] == row["gold"]:
            mark = "exact"
            exact += 1
            coarse += 1
        elif covers(result["label"], row["gold"]):
            mark = "coarse"
            coarse += 1
        else:
            mark = "miss"
        print(f"{mark:6} gold {row['gold']:12} label {result['label']:12} "
              f"leaf {result['leaf']:12} path_p={result['path_p']:.3f} "
              f"leaf_conf={result['confidence']}")
    print(f"\nexact {exact}/{len(DATA)}, gold or an ancestor of gold {coarse}/{len(DATA)}, "
          f"beam {BEAM}, confidence cutoff {MIN_CONFIDENCE}")


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
