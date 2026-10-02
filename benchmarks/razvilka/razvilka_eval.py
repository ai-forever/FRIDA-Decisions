"""Scorer for Razvilka, a Russian benchmark of structured decisions over text.

Standard library only; `datasets` is needed only to load the set from the Hub.

    import razvilka_eval as rz
    items = rz.load()                       # Hub, or rz.load("data/test.jsonl")
    result = rz.score(predictions, items)   # predictions: item id -> answer
    print(rz.format_report(result))

    python razvilka_eval.py --predictions predictions.jsonl [--data data/test.jsonl]

A prediction for one item may be a decided answer or a distribution:

    choice, ranking   an option key; {key: probability or any score}; or a list
                      of keys, best first (only the first one is used)
    noul              p(true) as a float in [0, 1]; a bool; "true" / "false";
                      or {"true": p, "false": q}
    score             a level index (int); a list of per-level probabilities;
                      or {level: probability}

An answer object as returned by FRIDA-Decisions (a dict with "type" and
"probabilities" / "noul") is accepted as is; its probabilities are used.

Decision rule: the argmax of the distribution, ties broken lexicographically by
option key (Python string order; score levels are the keys "0", "1", ...).
For noul this means true wins only when p(true) > 0.5. Keys left out of a
distribution cannot be chosen. Accuracy is counted over all items: an item with
no prediction, or a decided answer that is not one of its options, is wrong.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

HF_REPO = "artemsnegirev/razvilka"
TYPES = ("choice", "noul", "score", "ranking")
LOCAL_DATA = Path(__file__).resolve().parent / "data" / "test.jsonl"


# --------------------------------------------------------------------- data
def _parse(row: dict) -> dict:
    item = dict(row)
    for field in ("request", "gold"):
        if isinstance(item.get(field), str):
            item[field] = json.loads(item[field])
    return item


def load(path: str | Path | None = None, split: str = "test") -> list[dict]:
    """Items with `request` and `gold` decoded from JSON.

    `path` is a local `test.jsonl`; without it the set is loaded from the Hub
    with `datasets`.
    """
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            return [_parse(json.loads(line)) for line in handle if line.strip()]
    from datasets import load_dataset

    return [_parse(row) for row in load_dataset(HF_REPO, split=split)]


def question(item: dict) -> tuple[str, dict]:
    """(question id, question) -- every item asks exactly one question."""
    questions = item["request"]["questions"]
    if len(questions) != 1:
        raise ValueError(f"{item['id']}: expected one question, got {len(questions)}")
    return next(iter(questions.items()))


def options(item: dict) -> list[str]:
    """The option keys of an item, in request order."""
    _, q = question(item)
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    criteria = q["criteria"]
    if isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))]
    return [str(key) for key in criteria]


def gold_key(item: dict) -> str:
    gold = item["gold"]
    if item["type"] == "noul":
        return "true" if gold else "false"
    return str(gold)


# ---------------------------------------------------------------- decisions
def argmax(scores: Mapping[str, float]) -> str:
    """The key holding the maximum; ties go to the lexicographically first key."""
    best = max(scores.values())
    for key in sorted(scores):
        if scores[key] == best:
            return key
    raise ValueError(f"no key holds the maximum {best!r}")  # NaN in the scores


def _distribution(item: dict, values: Mapping) -> dict[str, float]:
    keys = set(options(item))
    out = {}
    for key, value in values.items():
        key = "true" if key is True else "false" if key is False else str(key)
        if key not in keys:
            raise ValueError(f"{item['id']}: option {key!r} is not one of {sorted(keys)}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{item['id']}: score for {key!r} is not a number: {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{item['id']}: score for {key!r} is not finite")
        out[key] = float(value)
    if not out:
        raise ValueError(f"{item['id']}: empty distribution")
    return out


def decide(item: dict, prediction) -> str | None:
    """The option key a prediction picks, or None when it picks no valid option."""
    if prediction is None:
        return None
    kind = item["type"]
    keys = options(item)

    # An answer object: {"type": ..., "probabilities": {...}} or {"type": "noul", "noul": p}.
    if isinstance(prediction, Mapping) and isinstance(prediction.get("type"), str):
        if kind == "noul" and "noul" in prediction:
            prediction = prediction["noul"]
        elif "probabilities" in prediction:
            prediction = prediction["probabilities"]
        elif kind in prediction:
            prediction = prediction[kind]
        else:
            raise ValueError(f"{item['id']}: answer object without probabilities")

    if kind == "noul":
        if isinstance(prediction, bool):
            return "true" if prediction else "false"
        if isinstance(prediction, str):
            word = prediction.strip().lower()
            return {"true": "true", "false": "false", "yes": "true", "no": "false"}.get(word)
        if isinstance(prediction, (int, float)):
            p = float(prediction)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{item['id']}: p(true) must be in [0, 1], got {p}")
            prediction = {"true": p, "false": 1.0 - p}
        if isinstance(prediction, Mapping):
            return argmax(_distribution(item, prediction))
        raise ValueError(f"{item['id']}: cannot read a noul prediction from {prediction!r}")

    if kind == "score":
        if isinstance(prediction, bool):
            raise ValueError(f"{item['id']}: a bool is not a score level")
        if isinstance(prediction, float) and prediction.is_integer():
            prediction = int(prediction)
        if isinstance(prediction, int):
            prediction = str(prediction)
        if isinstance(prediction, str):
            key = prediction.strip()
            return key if key in keys else None
        if isinstance(prediction, (list, tuple)):
            if len(prediction) != len(keys):
                raise ValueError(f"{item['id']}: {len(prediction)} probabilities "
                                 f"for {len(keys)} levels")
            prediction = dict(zip(keys, prediction))
        if isinstance(prediction, Mapping):
            return argmax(_distribution(item, prediction))
        raise ValueError(f"{item['id']}: cannot read a score prediction from {prediction!r}")

    # choice, ranking
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            return None
        prediction = prediction[0]
    if isinstance(prediction, (str, int)) and not isinstance(prediction, bool):
        key = str(prediction).strip()
        return key if key in keys else None
    if isinstance(prediction, Mapping):
        return argmax(_distribution(item, prediction))
    raise ValueError(f"{item['id']}: cannot read a {kind} prediction from {prediction!r}")


# ------------------------------------------------------------------ scoring
def _as_mapping(predictions) -> dict:
    if isinstance(predictions, Mapping):
        return dict(predictions)
    out = {}
    for record in predictions:
        if record["id"] in out:
            raise ValueError(f"duplicate prediction for {record['id']}")
        out[record["id"]] = record["prediction"]
    return out


def _block() -> dict:
    return {"n": 0, "correct": 0, "chance": 0.0, "missing": 0, "invalid": 0}


def _close(block: dict) -> dict:
    n = max(block["n"], 1)
    block["accuracy"] = block["correct"] / n
    block["chance"] = block["chance"] / n
    return block


def score(predictions: Mapping | Iterable[dict], items: list[dict] | None = None) -> dict:
    """Accuracy overall, per question type and per task, beside uniform chance.

    `predictions` maps item id -> prediction (or is an iterable of
    {"id", "prediction"} records). `items` defaults to the local
    `data/test.jsonl` when it exists, else the Hub.
    """
    if items is None:
        items = load(LOCAL_DATA if LOCAL_DATA.exists() else None)
    predictions = _as_mapping(predictions)
    known = {item["id"] for item in items}
    unknown = sorted(set(predictions) - known)
    if unknown:
        raise ValueError(f"{len(unknown)} prediction id(s) are not items of this set, "
                         f"e.g. {unknown[:3]}")

    total, per_type, per_task, records = _block(), {}, {}, []
    for item in items:
        prediction = predictions.get(item["id"])
        picked = decide(item, prediction)
        gold = gold_key(item)
        hit = int(picked == gold)
        blocks = (total, per_type.setdefault(item["type"], _block()),
                  per_task.setdefault(item["task"], {"type": item["type"], **_block()}))
        for block in blocks:
            block["n"] += 1
            block["correct"] += hit
            block["chance"] += 1.0 / len(options(item))
            block["missing"] += int(prediction is None)
            block["invalid"] += int(prediction is not None and picked is None)
        records.append({"id": item["id"], "task": item["task"], "type": item["type"],
                        "gold": gold, "predicted": picked, "correct": hit})

    _close(total)
    return {**total,
            "per_type": {t: _close(per_type[t]) for t in TYPES if t in per_type},
            "per_task": {t: _close(per_task[t]) for t in sorted(per_task)},
            "items": records}


def format_report(result: dict) -> str:
    """A plain-text table of a `score` result."""
    lines = [f"{'':<18} {'type':<8} {'n':>4} {'correct':>8} {'accuracy':>9} {'chance':>7}"]

    def row(name, kind, block):
        lines.append(f"{name:<18} {kind:<8} {block['n']:>4} {block['correct']:>8} "
                     f"{block['accuracy']:>9.4f} {block['chance']:>7.4f}")

    row("all", "", result)
    lines.append("")
    for kind, block in result["per_type"].items():
        row(kind, "", block)
    lines.append("")
    for task, block in result["per_task"].items():
        row(task, block["type"], block)
    if result["missing"] or result["invalid"]:
        lines.append(f"\n{result['missing']} item(s) without a prediction and "
                     f"{result['invalid']} invalid answer(s), all counted as wrong")
    return "\n".join(lines)


# --------------------------------------------------------- lexical baseline
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _render(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tokens(value) -> set[str]:
    return set(_TOKEN.findall(_render(value).lower()))


def _candidates(item: dict) -> list[tuple[str, str]]:
    _, q = question(item)
    criteria = q.get("criteria")
    if q["type"] == "noul":
        criteria = criteria or {}
        return [("true", "yes " + _render(criteria.get("true", ""))),
                ("false", "no " + _render(criteria.get("false", "")))]
    if q["type"] == "score":
        return [(str(i), _render(text)) for i, text in enumerate(criteria)]
    if isinstance(criteria, list):
        criteria = {str(i): text for i, text in enumerate(criteria)}
    if q["type"] == "ranking":
        return [(str(key), _render(text)) for key, text in criteria.items()]
    return [(str(key), str(key) if text is None else f"{key}: {_render(text)}")
            for key, text in criteria.items()]


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(x - top) for x in values]
    total = math.fsum(weights)
    return [w / total for w in weights]


def lexical_baseline(items: list[dict]) -> dict:
    """A model-free floor: IDF-weighted word overlap between the text and each option.

    An option's score is the sum of IDF over the words it shares with the text,
    divided by sqrt(number of distinct words in the option). IDF is counted over
    the given items' texts and option texts. Returns predictions for `score`.
    """
    candidates = [_candidates(item) for item in items]
    documents = [_tokens(item["request"]["state"]) for item in items]
    documents += [_tokens(text) for row in candidates for _, text in row]
    df = Counter(token for document in documents for token in document)
    total = len(documents)

    def idf(token):
        return math.log((total + 1) / (df.get(token, 0) + 1)) + 1.0

    predictions = {}
    for item, row in zip(items, candidates):
        state = _tokens(item["request"]["state"])
        margins = []
        for _, text in row:
            tokens = _tokens(text)
            margins.append(sum(idf(t) for t in tokens if t in state) / math.sqrt(len(tokens))
                           if tokens else 0.0)
        keys = [key for key, _ in row]
        if item["type"] == "noul":
            margin = margins[keys.index("true")] - margins[keys.index("false")]
            predictions[item["id"]] = (1 / (1 + math.exp(-margin)) if margin >= 0
                                       else math.exp(margin) / (1 + math.exp(margin)))
        else:
            predictions[item["id"]] = dict(zip(keys, _softmax(margins)))
    return predictions


# ---------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score predictions on Razvilka.")
    parser.add_argument("--predictions", required=True,
                        help='JSONL, one {"id": ..., "prediction": ...} per line')
    parser.add_argument("--data", default=None,
                        help="local test.jsonl (default: next to this file, else the Hub)")
    parser.add_argument("--json", default=None, help="also write the full result here")
    args = parser.parse_args(argv)

    data = args.data or (LOCAL_DATA if LOCAL_DATA.exists() else None)
    items = load(data)
    with open(args.predictions, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    result = score(records, items)
    print(format_report(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
