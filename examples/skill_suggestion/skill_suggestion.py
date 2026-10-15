"""Pick at most one intent from a catalog: rank, then re-check the top candidates.

    python examples/skill_suggestion/skill_suggestion.py
    python examples/skill_suggestion/skill_suggestion.py --device cpu
    python examples/skill_suggestion/skill_suggestion.py --onnx
    python examples/skill_suggestion/skill_suggestion.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
TOP_K = 5
NOUL_MIN = 0.5
FAST_POOL = 40


def pool_for(criteria: dict, gold: list[str], fast: bool) -> dict:
    if not fast:
        return criteria
    chosen = list(dict.fromkeys([*gold, *criteria]))[:FAST_POOL]
    return {key: criteria[key] for key in chosen}


def suggest(judge, text: str, criteria: dict) -> dict:
    ranked = judge({
        "state": text,
        "questions": {"best": {
            "type": "ranking",
            "instructions": "Какое намерение лучше всего описывает сообщение клиента?",
            "criteria": criteria,
        }},
    })
    top = ranked["answers"]["best"]["ranking"][:TOP_K]
    checked = judge({
        "state": text,
        "questions": {
            key: {"type": "noul", "instructions": "Клиент хочет именно это: " + criteria[key]}
            for key in top
        },
    })
    # A near-tie keeps the ranking's order. int8 ONNX can swap two 0.98s.
    best_noul = max(checked["answers"][key]["noul"] for key in top)
    close = [key for key in top if best_noul - checked["answers"][key]["noul"] <= 0.02]
    best = close[0]
    noul = checked["answers"][best]["noul"]
    return {
        "intent": best if noul >= NOUL_MIN else None,
        "noul": noul,
        "top": [(key, checked["answers"][key]["noul"]) for key in top],
        "truncated": ranked["usage"]["state_truncated"] or checked["usage"]["state_truncated"],
        "ranked_ms": ranked["usage"]["milliseconds"],
    }

# %% [markdown]
# ## Rank, then re-check
#
# The ranking always names a winner, because its probabilities split one unit across the list.
# The yes/no pass can refuse all of them. With `fast=True` the pool is 40 intents, not 243.

# %%
def run(judge, rows: list[dict], catalog_path, fast: bool = True) -> None:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    criteria = catalog["request"]["questions"]["intent"]["criteria"]
    picked = pool_for(criteria, [row["intent"] for row in rows], fast)
    print(f"pool {len(picked)} of {len(criteria)} intents, top-{TOP_K}, noul cutoff {NOUL_MIN}")
    hits = 0
    for row in rows:
        result = suggest(judge, row["text"], picked)
        hits += result["intent"] == row["intent"]
        top = " ".join(f"{key}:{noul:.2f}" for key, noul in result["top"])
        print(f"{'ok' if result['intent'] == row['intent'] else 'miss':4} "
              f"got {result['intent']!s:28} gold {row['intent']:28} "
              f"noul {result['noul']:.2f}  {result['ranked_ms']:.0f} ms")
        print(f"     top {top}")
        if result["truncated"]:
            print("     state truncated: default cut is 384 tokens")
    print(f"\n{hits}/{len(rows)} match. fast={fast}")
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
    parser.add_argument("--full", action="store_true",
                        help="rank all 243 intents; default is a pool of 40")
    args = parser.parse_args()
    judge = load_judge(args)
    catalog = Path(__file__).resolve().parents[1] / "data" / "intent_catalog.json"
    run(judge, load_rows(), catalog, fast=not args.full)


if __name__ == "__main__":
    main()
