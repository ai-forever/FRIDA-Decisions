"""The async API: `await judge.judge(request)`, with many requests in flight.

    python examples/async_judge.py                      # AsyncJudge over PyTorch, GPU if available
    python examples/async_judge.py --backend onnx       # AsyncJudge over int8 ONNX on CPU
    python examples/async_judge.py --backend mlx        # AsyncJudge over MLX on Apple Silicon
    python examples/async_judge.py --backend vllm       # VllmJudge: the vLLM engine in this process (Linux, GPU)
    python examples/async_judge.py --model path/to/exported/folder

Every variant takes the request `Judge` takes and returns the response `Judge`
returns; requests awaited at the same time are batched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from frida_decisions import DEFAULT_REPO_ID

TICKET = {
    "state": "Добрый день. Третий день не могу войти в личный кабинет: пишет, что пароль неверный, "
             "а письмо для сброса не приходит. Из-за этого не могу скачать счёт, срок оплаты завтра.",
    "questions": {
        "team": {"type": "choice", "instructions": "В какую команду направить обращение?",
                 "criteria": {"auth": "доступ к аккаунту, вход, пароли",
                              "billing": "счета, оплата, возвраты",
                              "shipping": "доставка заказов"}},
        "angry": {"type": "noul", "instructions": "Автор раздражён?"},
    },
}


def load(backend: str, model: str, gpu_memory_utilization: float):
    if backend == "vllm":
        from frida_decisions import VllmJudge
        return VllmJudge.from_pretrained(model, gpu_memory_utilization=gpu_memory_utilization)
    from frida_decisions import AsyncJudge
    return AsyncJudge.from_pretrained(model, backend=backend)


async def main(args) -> None:
    async with load(args.backend, args.model, args.gpu_memory_utilization) as judge:
        response = await judge.judge(TICKET)
        for qid, answer in response["answers"].items():
            print(qid, json.dumps(answer, ensure_ascii=False))

        # Many users at once: each coroutine awaits its own answer.
        requests = [dict(TICKET, state=f"Обращение №{i}. " + TICKET["state"]) for i in range(args.n)]
        started = time.perf_counter()
        responses = await asyncio.gather(*(judge.judge(r) for r in requests))
        seconds = time.perf_counter() - started
        teams = {r["answers"]["team"]["choice"] for r in responses}
        print(f"\n{args.n} requests at once: {seconds:.2f} s, {args.n / seconds:.0f} requests/s, teams {teams}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "onnx", "mlx", "vllm"], default="torch")
    parser.add_argument("--model", default=DEFAULT_REPO_ID)
    parser.add_argument("--n", type=int, default=100, help="requests awaited at once")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vllm only: lower it on a GPU that also drives a display")
    asyncio.run(main(parser.parse_args()))
