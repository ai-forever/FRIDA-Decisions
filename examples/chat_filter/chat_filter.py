"""Hide, blur or highlight chat messages with rules in code and AsyncJudge.

    python examples/chat_filter/chat_filter.py
    python examples/chat_filter/chat_filter.py --device cpu
    python examples/chat_filter/chat_filter.py --onnx
    python examples/chat_filter/chat_filter.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
import asyncio
import time

# First match wins. A question that is also spam is hidden, not highlighted.
# Question-ness is weak on its own, so a highlight also requires a question mark.
HIDE_INSULT = 0.70
HIDE_SPAM = 0.60
BLUR_SPOILER = 0.75
COLLAPSE_TIP = 0.70
HIGHLIGHT_QUESTION = 0.35


def request_for(text: str) -> dict:
    return {
        "state": text,
        "questions": {
            "insult": {"type": "noul", "instructions": "В сообщении оскорбляют человека?"},
            "spam": {"type": "noul", "instructions": "Это реклама или навязчивый призыв купить?"},
            "spoiler": {"type": "noul", "instructions": "Сообщение заранее раскрывает сюжет или развязку?"},
            "tip": {"type": "noul", "instructions": "Это непрошеный совет, как играть или что настроить?"},
            "question": {"type": "noul", "instructions": "Автор задаёт вопрос другим участникам?"},
        },
    }


def moderate(text: str, answers: dict) -> str:
    if answers["insult"]["noul"] >= HIDE_INSULT or answers["spam"]["noul"] >= HIDE_SPAM:
        return "hide"
    if answers["spoiler"]["noul"] >= BLUR_SPOILER:
        return "blur"
    if answers["tip"]["noul"] >= COLLAPSE_TIP:
        return "collapse"
    if "?" in text and answers["question"]["noul"] >= HIGHLIGHT_QUESTION:
        return "highlight"
    return "show"

# %% [markdown]
# ## Several messages at once
#
# `asyncio.gather` awaits every message together. `AsyncJudge` scores that group in one `judge_batch` on the worker thread, so the event loop stays free while the model runs.

# %%
async def run(judge, rows: list[dict]) -> None:
    started = time.perf_counter()
    responses = await asyncio.gather(*(judge.judge(request_for(row["text"])) for row in rows))
    elapsed = time.perf_counter() - started
    hits = 0
    for row, response in zip(rows, responses):
        action = moderate(row["text"], response["answers"])
        hits += action == row["action"]
        probs = " ".join(f"{name}={response['answers'][name]['noul']:.2f}"
                         for name in ("insult", "spam", "spoiler", "tip", "question"))
        print(f"{'ok' if action == row['action'] else 'miss':4} got {action:10} gold {row['action']:10} {probs}")
        print(f"     {row['text']}")
    print(f"\n{hits}/{len(rows)} actions match, {len(rows)} messages in {elapsed:.2f}s")
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
    import asyncio

    from frida_decisions import AsyncJudge

    async def _go():
        if args.onnx:
            opened = AsyncJudge.from_pretrained(args.model, backend="onnx", threads=args.threads)
        else:
            import torch
            if args.threads:
                torch.set_num_threads(args.threads)
            opened = AsyncJudge.from_pretrained(args.model, backend="torch", device=args.device)
        async with opened as judge:
            await run(judge, load_rows())

    asyncio.run(_go())


if __name__ == "__main__":
    main()
