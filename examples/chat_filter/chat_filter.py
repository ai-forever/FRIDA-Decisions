"""Hide, blur or highlight chat messages with rules in code and AsyncJudge.

    python examples/chat_filter/chat_filter.py
    python examples/chat_filter/chat_filter.py --device cpu
    python examples/chat_filter/chat_filter.py --onnx
    python examples/chat_filter/chat_filter.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse

from frida_decisions import DEFAULT_REPO_ID

import asyncio
import time

DATA = [
    {"action": "highlight", "text": "Подскажите, пожалуйста, где крафтится верстак?"},
    {"action": "hide", "text": "Купите золото дёшево, пишите в личные сообщения, лучшая цена на сервере!!!"},
    {"action": "hide", "text": "Ты тупица и играешь отвратительно."},
    {"action": "blur", "text": "Осторожно, спойлер: босс умирает с одного удара в глаз."},
    {"action": "collapse", "text": "Поставь чувствительность мыши на 800, так лучше целятся."},
    {"action": "show", "text": "Спасибо за игру, до завтра."},
    {"action": "blur", "text": "Спойлер пятой серии: король оказывается предателем."},
    {"action": "highlight", "text": "Есть свободное место в группе на вечер?"},
    {"action": "hide", "text": "Подскажи, куда жать, и кстати купи золото у меня в личке, дёшево."},
]
HIDE_INSULT = 0.70
HIDE_SPAM = 0.60
BLUR_SPOILER = 0.75
COLLAPSE_TIP = 0.70
HIGHLIGHT_QUESTION = 0.35


def request_for(text):
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


def moderate(text, answers):
    if answers["insult"]["noul"] >= HIDE_INSULT or answers["spam"]["noul"] >= HIDE_SPAM:
        return "hide"
    if answers["spoiler"]["noul"] >= BLUR_SPOILER:
        return "blur"
    if answers["tip"]["noul"] >= COLLAPSE_TIP:
        return "collapse"
    if "?" in text and answers["question"]["noul"] >= HIGHLIGHT_QUESTION:
        return "highlight"
    return "show"


async def run(judge):
    started = time.perf_counter()
    responses = await asyncio.gather(*(judge.judge(request_for(row["text"])) for row in DATA))
    elapsed = time.perf_counter() - started
    hits = 0
    for row, response in zip(DATA, responses):
        action = moderate(row["text"], response["answers"])
        if action == row["action"]:
            mark = "ok"
            hits += 1
        else:
            mark = "miss"
        probs = " ".join(
            f"{name}={response['answers'][name]['noul']:.2f}"
            for name in ("insult", "spam", "spoiler", "tip", "question")
        )
        print(f"{mark:4} got {action:10} gold {row['action']:10} {probs}")
        print(f"     {row['text']}")
    print(f"\n{hits}/{len(DATA)} actions match, {len(DATA)} messages in {elapsed:.2f}s")


async def main_async(args):
    from frida_decisions import AsyncJudge
    if args.onnx:
        opened = AsyncJudge.from_pretrained(args.model, backend="onnx")
    else:
        opened = AsyncJudge.from_pretrained(args.model, backend="torch", device=args.device)
    async with opened as judge:
        await run(judge)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default=None, help="cuda or cpu (torch backend)")
    parser.add_argument("--onnx", action="store_true", help="int8 ONNX on CPU")

    args = parser.parse_args()
    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
