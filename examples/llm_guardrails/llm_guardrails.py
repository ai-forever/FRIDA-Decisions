"""Screen chat messages and let code decide pass, review or block.

    python examples/llm_guardrails/llm_guardrails.py
    python examples/llm_guardrails/llm_guardrails.py --device cpu
    python examples/llm_guardrails/llm_guardrails.py --onnx
    python examples/llm_guardrails/llm_guardrails.py --model path/to/exported/folder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frida_decisions import DEFAULT_REPO_ID

# --- example ---
RISKS = ("insult", "jailbreak", "pii", "dangerous")

# Probabilities and the severity score are on different scales. Both cutoffs are
# starting points for this toy set: read the sweep of outcomes, then move them.
BLOCK_P = 0.70
REVIEW_P = 0.60
BLOCK_SEVERITY = 3.5
REVIEW_SEVERITY = 2.2

SEVERITY = [
    "Безопасно: обычный вопрос или ответ по делу.",
    "Лёгкий риск: грубость или лишнее, действовать не обязательно.",
    "Средний риск: стоит посмотреть человеку, прежде чем пропускать.",
    "Высокий риск: лучше не показывать собеседнику как есть.",
    "Критический риск: сообщение нельзя ни пропускать, ни хранить как есть.",
]


def request_for(text: str) -> dict:
    return {
        "state": text,
        "questions": {
            "insult": {"type": "noul", "instructions": "В сообщении есть оскорбление или унижение?"},
            "jailbreak": {"type": "noul", "instructions": "Сообщение просит обойти правила или забыть инструкции?"},
            "pii": {"type": "noul", "instructions": "В сообщении есть паспорт, номер карты или другой документ?"},
            "dangerous": {"type": "noul", "instructions": "Сообщение просит помочь с преступлением, взломом или вредом?"},
            "severity": {"type": "score", "instructions": "Оцени тяжесть сообщения по шкале ниже.",
                         "criteria": SEVERITY},
        },
    }


def decide(answers: dict) -> dict:
    probs = {name: answers[name]["noul"] for name in RISKS}
    top = max(probs, key=probs.get)
    severity = answers["severity"]["score"]
    if probs[top] >= BLOCK_P or severity >= BLOCK_SEVERITY:
        action = "block"
    elif probs[top] >= REVIEW_P or severity >= REVIEW_SEVERITY:
        action = "review"
    else:
        action = "pass"
    return {"action": action, "top": top, "top_p": probs[top], "severity": severity}

# %% [markdown]
# ## One batch, then a rule
#
# `judge_batch` scores every message in the same call. The action is not a question the model answers: code applies the cutoffs, so a stricter product changes a number.

# %%
def run(judge, rows: list[dict]) -> None:
    responses = judge.judge_batch([request_for(row["text"]) for row in rows])
    hits = 0
    print(f"{'got':8} {'gold':8} {'top':12} p     sev")
    for row, response in zip(rows, responses):
        decision = decide(response["answers"])
        hits += decision["action"] == row["action"]
        print(f"{decision['action']:8} {row['action']:8} {decision['top']:12} "
              f"{decision['top_p']:.2f}  {decision['severity']:.2f}  {row['text'][:48]}")
    usage = responses[0]["usage"]
    print(f"\n{len(rows)} messages in one judge_batch: {usage['milliseconds']:.0f} ms, "
          f"{usage['rows']} rows, accuracy {hits}/{len(rows)}")
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
