"""Pick at most one intent from a catalog: rank, then re-check the top candidates.

    python examples/skill_suggestion/skill_suggestion.py
    python examples/skill_suggestion/skill_suggestion.py --device cpu
    python examples/skill_suggestion/skill_suggestion.py --onnx
    python examples/skill_suggestion/skill_suggestion.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

import json
import urllib.request
from pathlib import Path

DATA = [
    {"text": "Я клиент другого оператора. Хочу стать вашим абонентом и перенести к вам свой текущий номер.", "intent": "port_in_keep_number"},
    {"text": "Потерял телефон в метро, заблокируйте сим-карту, чтобы с неё никто не звонил.", "intent": "sim_lost_block"},
    {"text": "Какой у меня сейчас тариф и что в него входит?", "intent": "tariff_current_info"},
    {"text": "Хочу заменить пластиковую сим-карту на eSIM.", "intent": "esim_activate"},
]
TOP_K = 5
NOUL_MIN = 0.5
TIE = 0.02
FAST_POOL = 40
CATALOG_URL = ("https://raw.githubusercontent.com/ai-forever/FRIDA-Decisions/"
               "main/examples/data/intent_catalog.json")


def load_catalog():
    for path in (Path("examples/data/intent_catalog.json"), Path("../data/intent_catalog.json")):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(CATALOG_URL, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pool_for(criteria, fast):
    if not fast:
        return criteria
    gold = [row["intent"] for row in DATA]
    chosen = []
    for key in gold + list(criteria):
        if key not in chosen:
            chosen.append(key)
        if len(chosen) == FAST_POOL:
            break
    picked = {}
    for key in chosen:
        picked[key] = criteria[key]
    return picked


def suggest(judge, text, criteria):
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
    best_noul = 0.0
    for key in top:
        best_noul = max(best_noul, checked["answers"][key]["noul"])
    best = None
    for key in top:
        if best_noul - checked["answers"][key]["noul"] <= TIE:
            best = key
            break
    noul = checked["answers"][best]["noul"]
    intent = None
    if noul >= NOUL_MIN:
        intent = best
    top_rows = []
    for key in top:
        top_rows.append((key, checked["answers"][key]["noul"]))
    truncated = ranked["usage"]["state_truncated"] or checked["usage"]["state_truncated"]
    return {"intent": intent, "noul": noul, "top": top_rows,
            "truncated": truncated, "ranked_ms": ranked["usage"]["milliseconds"]}


def run(judge, fast=True):
    catalog = load_catalog()
    criteria = catalog["request"]["questions"]["intent"]["criteria"]
    picked = pool_for(criteria, fast)
    print(f"pool {len(picked)} of {len(criteria)} intents, top-{TOP_K}, noul cutoff {NOUL_MIN}")
    hits = 0
    for row in DATA:
        result = suggest(judge, row["text"], picked)
        if result["intent"] == row["intent"]:
            mark = "ok"
            hits += 1
        else:
            mark = "miss"
        shown = " ".join(f"{key}:{noul:.2f}" for key, noul in result["top"])
        print(f"{mark:4} got {result['intent']!s:28} gold {row['intent']:28} "
              f"noul {result['noul']:.2f}  {result['ranked_ms']:.0f} ms")
        print(f"     top {shown}")
        if result["truncated"]:
            print("     state truncated: default cut is 384 tokens")
    print(f"\n{hits}/{len(DATA)} match. fast={fast}")


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
    parser.add_argument("--full", action="store_true",
                        help="rank all 243 intents; default is a pool of 40")
    args = parser.parse_args()
    run(load_judge(args), fast=not args.full)


if __name__ == "__main__":
    main()
