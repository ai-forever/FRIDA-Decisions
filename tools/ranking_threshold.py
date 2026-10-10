"""Can one threshold on `ranking` margins decide relevance across requests?

`ranking` returns `scores`, one margin per candidate, computed independently of
the other candidates. Inside one request they order candidates; the question
here is whether their *level* carries over, so that a caller can keep a single
threshold and filter passages with it.

The head was trained with a listwise softmax, which depends only on differences
inside one list, so the level of a request's margins is not pinned by the loss.
This measures what that costs on the relevance task of razvilka (50 queries,
5 candidates each, one relevant and four annotated negatives), and which of
three ways of removing the offset actually works:

* as they are            -- what a single global threshold reaches;
* minus an anchor        -- one extra candidate with a fixed text is added to
                            every request and the threshold is put on the
                            distance to it. Measured here because it is the
                            first thing one reaches for, and it does not work:
                            the anchor's own margin barely moves with the
                            request (see the correlation this prints);
* minus the list         -- the mean, or the median, of the request's own
                            candidates. This does work, and needs nothing from
                            the caller but the candidates already in the list.

Each threshold is also scored held out -- fitted on four fifths of the requests
and read off the rest -- because the best single threshold is otherwise chosen
on the same candidates it is judged on. The last two checks are what the recipe
cannot do: the fitted threshold is re-applied to shorter lists, and to lists
with the relevant candidate removed, where centring has nothing to find and can
only promote the best of the rest.

    python tools/ranking_threshold.py [model_dir] [--data path/to/test.jsonl]
    python tools/ranking_threshold.py --rows tests/_results/ranking_threshold.json

Writes `tests/_results/ranking_threshold.json` with every margin; `--rows`
recomputes every number from it, with no model and no dataset.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "razvilka"))

import razvilka_eval as rz  # noqa: E402

from frida_decisions import Judge  # noqa: E402

TASK = "relevance_list"

# The anchor is one more candidate, with the same wording in every request, which
# says of itself that it is not an answer.
ANCHOR_ID = "__anchor__"
ANCHOR_TEXT = "Этот фрагмент не относится к запросу и не отвечает на него."


def auc(positives: list[float], negatives: list[float]) -> float:
    """Share of (positive, negative) pairs the positive wins; a tie counts half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def best_threshold(positives: list[float], negatives: list[float]) -> dict:
    """The single threshold with the highest F1, over every candidate value."""
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
    for threshold in sorted(set(positives + negatives)):
        true_positive = sum(x >= threshold for x in positives)
        false_positive = sum(x >= threshold for x in negatives)
        if not true_positive:
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / len(positives)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": precision, "recall": recall,
                    "threshold": threshold}
    return best


def split(rows: list[dict], shift) -> tuple[list[float], list[float]]:
    positives = [row["scores"][row["gold"]] - shift(row) for row in rows]
    negatives = [value - shift(row) for row in rows
                 for key, value in row["scores"].items() if key != row["gold"]]
    return positives, negatives


def f1_at(threshold: float, positives: list[float], negatives: list[float]) -> float:
    true_positive = sum(x >= threshold for x in positives)
    if not true_positive:
        return 0.0
    precision = true_positive / (true_positive + sum(x >= threshold for x in negatives))
    recall = true_positive / len(positives)
    return 2 * precision * recall / (precision + recall)


def cross_validated(rows: list[dict], shift, folds: int = 5, repeats: int = 200,
                    seed: int = 0) -> float:
    """Mean held-out F1 when the threshold is fitted on the other requests.

    The best single threshold is read off the same candidates it is scored on,
    which flatters it. Splitting by request — never by candidate, or a request's
    own candidates would sit on both sides — says what it is worth on requests
    it was not fitted to.
    """
    generator = random.Random(seed)
    order = list(range(len(rows)))
    scores = []
    for _ in range(repeats):
        generator.shuffle(order)
        for fold in range(folds):
            held = {order[i] for i in range(fold, len(order), folds)}
            fit = [row for i, row in enumerate(rows) if i not in held]
            test = [row for i, row in enumerate(rows) if i in held]
            threshold = best_threshold(*split(fit, shift))["threshold"]
            scores.append(f1_at(threshold, *split(test, shift)))
    return statistics.mean(scores)


def bootstrap_auc(rows: list[dict], shift, draws: int = 2000, seed: int = 0) -> tuple:
    """95 % interval of the pooled AUC, resampling requests (50 is a small set)."""
    generator = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [rows[generator.randrange(len(rows))] for _ in rows]
        values.append(auc(*split(sample, shift)))
    values.sort()
    return values[int(0.025 * draws)], values[int(0.975 * draws)]


def shorter_lists(rows: list[dict], threshold: float) -> dict:
    """The same threshold on lists holding fewer candidates.

    Centring on the list is only as good as the list's estimate of where
    "irrelevant" sits, and the relevant candidate is itself part of the mean.
    Every subset of a request's negatives is used, so k = 1 means the relevant
    candidate against one negative.
    """
    out = {}
    for keep in range(1, 5):
        positives, negatives, fitted = [], [], []
        for row in rows:
            gold = row["gold"]
            pool = [(k, v) for k, v in row["scores"].items() if k != gold]
            for subset in itertools.combinations(pool, keep):
                values = [row["scores"][gold]] + [v for _, v in subset]
                level = statistics.mean(values)
                positives.append(row["scores"][gold] - level)
                negatives += [v - level for _, v in subset]
        fitted = best_threshold(positives, negatives)
        out[keep + 1] = {"auc": auc(positives, negatives),
                         "f1_refitted": fitted["f1"], "threshold_refitted": fitted["threshold"],
                         "f1_at_reference": f1_at(threshold, positives, negatives)}
    return out


def nothing_relevant(rows: list[dict], threshold: float, shift) -> dict:
    """How many candidates clear the threshold on lists with no relevant one.

    The relevant candidate is dropped and the rest are centred among
    themselves. Centring cannot tell "the best of this list" from "relevant",
    so this is the false-positive rate it leaves behind.
    """
    above, total = 0, 0
    for row in rows:
        scores = {k: v for k, v in row["scores"].items() if k != row["gold"]}
        stripped = {"scores": scores, "gold": None, "anchor": row["anchor"]}
        level = shift(stripped)
        above += sum(v - level >= threshold for v in scores.values())
        total += len(scores)
    return {"above": above, "candidates": total, "share": above / total}


def variant(rows: list[dict], shift) -> dict:
    """AUC and the best single threshold after subtracting `shift` per request."""
    positives, negatives = split(rows, shift)
    low, high = bootstrap_auc(rows, shift)
    return {"auc": auc(positives, negatives), "auc_ci95": [low, high],
            **best_threshold(positives, negatives),
            "f1_cross_validated": cross_validated(rows, shift)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=str(ROOT / "_export" / "FRIDA-Decisions"))
    parser.add_argument("--data", help="local razvilka test.jsonl (default: the Hub)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rows", help="recompute every number from a previous run's JSON, "
                                       "without loading the model")
    args = parser.parse_args(argv)

    if args.rows:
        saved = json.loads(Path(args.rows).read_text(encoding="utf-8"))
        return report(saved["rows"], saved["anchor_margin_drift"],
                      {"model": saved["model"], "backend": saved["backend"],
                       "device": saved["device"]}, Path(args.rows))

    items = [item for item in rz.load(args.data) if item["task"] == TASK]
    if not items:
        print(f"no {TASK} items in the set")
        return 1
    judge = Judge.from_pretrained(args.model, device=args.device)
    print(f"{len(items)} {TASK} requests, {judge.backend} on {judge.device}", flush=True)

    rows, drift = [], 0.0
    for item in items:
        qid, _ = rz.question(item)
        answer = judge(item["request"])["answers"][qid]

        # The same request with the anchor appended. A candidate's margin does
        # not depend on the rest of the list, so the other margins must come
        # back unchanged -- `drift` checks that rather than assuming it.
        anchored = json.loads(json.dumps(item["request"]))
        anchored["questions"][qid]["criteria"][ANCHOR_ID] = ANCHOR_TEXT
        with_anchor = judge(anchored)["answers"][qid]["scores"]
        drift = max(drift, max(abs(with_anchor[k] - v)
                               for k, v in answer["scores"].items()))

        rows.append({"id": item["id"], "gold": rz.gold_key(item),
                     "top1": answer["ranking"][0], "scores": answer["scores"],
                     "anchor": with_anchor[ANCHOR_ID]})
    print(f"margins unchanged by the anchor to {drift:.2g}", flush=True)
    return report(rows, drift, {"model": Path(args.model).name, "backend": judge.backend,
                                "device": str(judge.device)})


def report(rows: list[dict], drift: float, about: dict, source: Path | None = None) -> int:
    """Everything the docs quote, computed from the margins alone."""
    if source is not None:
        print(f"{len(rows)} requests from {source.name} "
              f"({about['backend']} on {about['device']})", flush=True)

    hits = sum(row["top1"] == row["gold"] for row in rows)
    levels = [statistics.mean(row["scores"].values()) for row in rows]
    anchors = [row["anchor"] for row in rows]
    positives = [row["scores"][row["gold"]] for row in rows]
    negatives = [v for row in rows for k, v in row["scores"].items() if k != row["gold"]]
    gaps = [row["scores"][row["gold"]] - max(v for k, v in row["scores"].items()
                                             if k != row["gold"]) for row in rows]

    variants = {
        "raw": variant(rows, lambda row: 0.0),
        "minus_anchor": variant(rows, lambda row: row["anchor"]),
        "minus_list_mean": variant(rows, lambda row: statistics.mean(row["scores"].values())),
        "minus_list_median": variant(rows, lambda row: statistics.median(row["scores"].values())),
    }
    # The recipe the docs recommend, and what it is blind to.
    mean_shift = lambda row: statistics.mean(row["scores"].values())    # noqa: E731
    reference = variants["minus_list_mean"]["threshold"]
    result = {
        **about, "requests": len(rows),
        "candidates_per_request": len(rows[0]["scores"]),
        "reference_threshold": reference,
        "shorter_lists": shorter_lists(rows, reference),
        "nothing_relevant_minus_list_mean": nothing_relevant(rows, reference, mean_shift),
        "nothing_relevant_raw": nothing_relevant(
            rows, variants["raw"]["threshold"], lambda row: 0.0),
        "top1": hits / len(rows),
        "auc_within_request": statistics.mean(
            auc([row["scores"][row["gold"]]],
                [v for k, v in row["scores"].items() if k != row["gold"]]) for row in rows),
        "variants": variants,
        "anchor_text": ANCHOR_TEXT, "anchor_margin_drift": drift,
        # Why the anchor cannot remove the offset: its own margin hardly moves
        # with the request, so subtracting it subtracts noise, not the offset.
        "anchor_level": {"median": statistics.median(anchors), "sd": statistics.pstdev(anchors),
                         "correlation_with_list_mean": statistics.correlation(anchors, levels)},
        "list_level": {"median": statistics.median(levels), "min": min(levels),
                       "max": max(levels), "sd": statistics.pstdev(levels)},
        "positive": {"median": statistics.median(positives), "min": min(positives),
                     "max": max(positives)},
        "negative": {"median": statistics.median(negatives), "min": min(negatives),
                     "max": max(negatives)},
        "gap_to_best_negative": {"median": statistics.median(gaps),
                                 "not_positive": sum(g <= 0 for g in gaps)},
        "rows": rows,
    }

    print(f"\ntop-1 within a request  {hits}/{len(rows)} = {result['top1']:.3f}")
    print(f"AUC within a request    {result['auc_within_request']:.3f}\n")
    for name, value in variants.items():
        low, high = value["auc_ci95"]
        print(f"{name:<18} AUC {value['auc']:.3f} [{low:.3f}, {high:.3f}]   "
              f"F1 {value['f1']:.3f} "
              f"(precision {value['precision']:.3f}, recall {value['recall']:.3f}) "
              f"at {value['threshold']:+.2f};  held-out F1 {value['f1_cross_validated']:.3f}")

    print(f"\nthe same threshold ({reference:+.2f}, minus the list mean) on shorter lists:")
    for size, value in sorted(result["shorter_lists"].items()):
        print(f"  {size} candidates  AUC {value['auc']:.3f}   "
              f"F1 {value['f1_at_reference']:.3f} at {reference:+.2f}, "
              f"{value['f1_refitted']:.3f} refitted at {value['threshold_refitted']:+.2f}")
    none_mean = result["nothing_relevant_minus_list_mean"]
    none_raw = result["nothing_relevant_raw"]
    print(f"\nlists with the relevant candidate removed: "
          f"{none_mean['above']}/{none_mean['candidates']} "
          f"({none_mean['share']:.1%}) still clear {reference:+.2f} after centring; "
          f"{none_raw['above']}/{none_raw['candidates']} ({none_raw['share']:.1%}) "
          f"clear {variants['raw']['threshold']:+.2f} raw")

    level, anchor = result["list_level"], result["anchor_level"]
    print(f"\nrequest level (mean margin) median {level['median']:+.2f}, "
          f"{level['min']:+.2f} .. {level['max']:+.2f}, sd {level['sd']:.2f}")
    print(f"anchor margin               median {anchor['median']:+.2f}, sd {anchor['sd']:.2f}, "
          f"correlation with the level {anchor['correlation_with_list_mean']:+.2f}")
    print(f"relevant   median {result['positive']['median']:+.2f}, "
          f"{result['positive']['min']:+.2f} .. {result['positive']['max']:+.2f}")
    print(f"irrelevant median {result['negative']['median']:+.2f}, "
          f"{result['negative']['min']:+.2f} .. {result['negative']['max']:+.2f}")
    print(f"gap to the best negative: median {result['gap_to_best_negative']['median']:+.2f}, "
          f"not positive in {result['gap_to_best_negative']['not_positive']} requests")

    out = ROOT / "tests" / "_results" / "ranking_threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
