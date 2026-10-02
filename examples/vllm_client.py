"""Ask a running vLLM server: a ticket, a follow-up about the same text, a 243-intent catalog, a burst.

Start the server first (`pip install "frida-decisions[vllm]"`, a Linux GPU machine):

    vllm serve ai-forever/FRIDA-Decisions \\
      --hf-overrides '{"architectures": ["FridaDecisionsModel"]}' \\
      --io-processor-plugin frida_decisions \\
      --no-enable-chunked-prefill --enforce-eager --max-model-len 2048

then, from anywhere (standard library only, no frida-decisions needed on the client):

    python examples/vllm_client.py                          # http://localhost:8000
    python examples/vllm_client.py --url http://gpu-host:8000

A request is the JSON `Judge` takes; the answer is what `Judge` returns, plus
`usage.cached_tokens`: how much of the text the server took from its cache.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

# Another question about the same text: the server reads the text from its cache.
FOLLOW_UP = {
    "state": TICKET["state"],
    "questions": {
        "urgent": {"type": "noul", "instructions": "Нужно ли ответить клиенту сегодня?"},
    },
}


class Client:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        # Requests to the server go direct, even when the machine has a proxy configured.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.opener.open(f"{self.url}/v1/models") as response:
            self.model = json.load(response)["data"][0]["id"]

    def ask(self, request: dict) -> dict:
        body = json.dumps({"model": self.model, "data": request}, ensure_ascii=False).encode()
        call = urllib.request.Request(f"{self.url}/pooling", body,
                                      {"Content-Type": "application/json"})
        with self.opener.open(call) as response:
            return json.load(response)["data"]


def timed(client: Client, request: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    response = client.ask(request)
    return response, (time.perf_counter() - started) * 1000


def show(label: str, response: dict, ms: float) -> None:
    usage = response["usage"]
    print(f"\n{label}: {ms:.0f} ms, {usage['rows']} row(s), {usage['prompt_tokens']} tokens, "
          f"{usage['cached_tokens']} from the cache")
    for qid, answer in response["answers"].items():
        print(" ", qid, json.dumps(answer, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--burst", type=int, default=64, help="requests in the throughput burst")
    parser.add_argument("--in-flight", type=int, default=8)
    args = parser.parse_args()
    client = Client(args.url)

    # Several questions about one text, answered together.
    show("ticket", *timed(client, TICKET))
    # A new question about a text the server has already read.
    show("follow-up, same text", *timed(client, FOLLOW_UP))

    catalog = json.loads((Path(__file__).parent / "data" / "intent_catalog.json")
                         .read_text(encoding="utf-8"))
    response, ms = timed(client, catalog["request"])
    answer = response["answers"]["intent"]
    print(f"\n243 intents -> {answer['choice']} (p={answer['probabilities'][answer['choice']]:.2f}, "
          f"expected {catalog['expected']['intent']}) in {ms:.0f} ms, "
          f"{response['usage']['rows']} rows, {response['usage']['cached_tokens']} tokens from the cache")

    # Many users at once: distinct texts, `--in-flight` requests at a time.
    requests = [dict(TICKET, state=f"Обращение №{i}. " + TICKET["state"]) for i in range(args.burst)]
    started = time.perf_counter()
    with ThreadPoolExecutor(args.in_flight) as pool:
        list(pool.map(client.ask, requests))
    seconds = time.perf_counter() - started
    print(f"\n{args.burst} requests, {args.in_flight} in flight: {args.burst / seconds:.0f} requests/s")


if __name__ == "__main__":
    main()
