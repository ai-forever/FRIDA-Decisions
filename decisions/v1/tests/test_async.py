"""(f) The async API: `AsyncJudge` (any synchronous judge) and `VllmJudge` (vLLM engine).

CPU only, no vLLM needed:

* `AsyncJudge` over a stand-in judge: requests awaited together are scored in
  shared `judge_batch` calls, every caller gets its own answer, an invalid
  request fails alone, a model failure reaches only the callers of that call,
  a cancelled caller does not block anyone, `close()` leaves no one waiting,
  and the event loop keeps running while the model computes;
* `AsyncJudge` over `Judge` (float32): the same margins as `Judge.judge`;
* `VllmJudge` over a stand-in engine that scores each row with the backend's
  own arithmetic (`kernel`) on a small random T5: the same margins as `Judge`;
  a cancelled request aborts its rows in the engine.
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from conftest import flat_margins, load_cases, record

from frida_decisions import AsyncJudge, RequestError
from frida_decisions.protocol import decision

REQUEST = {"state": "текст", "questions": {"q": {"type": "noul", "instructions": "Да?"}}}


class StandIn:
    """A synchronous judge that echoes its requests and records its calls."""

    backend = "stand-in"

    def __init__(self, delay: float = 0.0, fail_on: str | None = None):
        self.delay, self.fail_on = delay, fail_on
        self.calls: list[int] = []
        self.threads: set[str] = set()

    def judge_batch(self, requests, calibration=None):
        self.calls.append(len(requests))
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        if any(r["state"] == self.fail_on for r in requests):
            raise RuntimeError("model failure")
        return [{"echo": r["state"]} for r in requests]


def with_state(text: str) -> dict:
    return dict(REQUEST, state=text)


def test_concurrent_requests_share_calls_and_get_their_own_answers():
    async def main():
        inner = StandIn(delay=0.05)
        async with AsyncJudge(inner, max_batch=8) as judge:
            out = await asyncio.gather(*(judge.judge(with_state(f"t{i}")) for i in range(20)))
        return inner, out

    inner, out = asyncio.run(main())
    assert [o["echo"] for o in out] == [f"t{i}" for i in range(20)]
    assert sum(inner.calls) == 20 and max(inner.calls) <= 8 and len(inner.calls) < 20
    assert len(inner.threads) == 1                        # the model is used from one thread


def test_invalid_request_fails_alone():
    async def main():
        async with AsyncJudge(StandIn()) as judge:
            results = await asyncio.gather(judge.judge(with_state("ok")),
                                           judge.judge({"state": "x", "questions": {}}),
                                           return_exceptions=True)
        return results

    good, bad = asyncio.run(main())
    assert good == {"echo": "ok"} and isinstance(bad, RequestError)


def test_model_failure_reaches_only_its_call():
    async def main():
        inner = StandIn(delay=0.05, fail_on="boom")
        async with AsyncJudge(inner, max_batch=1) as judge:
            return await asyncio.gather(judge.judge(with_state("boom")), judge.judge(with_state("fine")),
                                        return_exceptions=True)

    boom, fine = asyncio.run(main())
    assert isinstance(boom, RuntimeError) and fine == {"echo": "fine"}


def test_cancelled_caller_does_not_block_others():
    async def main():
        async with AsyncJudge(StandIn(delay=0.1), max_batch=1) as judge:
            first = asyncio.create_task(judge.judge(with_state("a")))
            second = asyncio.create_task(judge.judge(with_state("b")))
            third = asyncio.create_task(judge.judge(with_state("c")))
            await asyncio.sleep(0.02)
            second.cancel()
            return await first, await third, second.cancelled()

    assert asyncio.run(main()) == ({"echo": "a"}, {"echo": "c"}, True)


def test_close_leaves_no_one_waiting():
    async def main():
        judge = AsyncJudge(StandIn(delay=0.2), max_batch=1)
        tasks = [asyncio.create_task(judge.judge(with_state(f"t{i}"))) for i in range(3)]
        await asyncio.sleep(0.05)                       # the first one is being scored
        await asyncio.wait_for(judge.close(), 5)
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        with pytest.raises(RuntimeError):
            await judge.judge(with_state("late"))
        return results

    results = asyncio.run(main())
    assert all(isinstance(r, RuntimeError) for r in results[1:])


def test_event_loop_runs_while_the_model_computes():
    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        async with AsyncJudge(StandIn(delay=0.3)) as judge:
            await judge.judge(with_state("slow"))
        task.cancel()
        return ticks

    assert asyncio.run(main()) >= 10


# ----------------------------------------------------------------- real Judge
def test_async_judge_matches_judge(torch_judge, cases):
    """All cases awaited at once (shared judge_batch calls) against one by one."""
    sequential = {c["name"]: torch_judge.judge(c["request"]) for c in cases}
    calls: list[int] = []
    original = torch_judge.judge_batch

    def recording(requests, calibration=None):
        calls.append(len(requests))
        return original(requests, calibration)

    async def main():
        torch_judge.judge_batch = recording
        try:
            async with AsyncJudge(torch_judge, max_batch=4) as judge:
                return await asyncio.gather(*(judge.judge(c["request"]) for c in cases))
        finally:
            del torch_judge.judge_batch

    out = asyncio.run(main())
    drift, mismatches = 0.0, []
    for case, response in zip(cases, out):
        ref = sequential[case["name"]]
        drift = max(drift, max(abs(a - b) for a, b in zip(flat_margins(ref), flat_margins(response))))
        if {q: decision(a) for q, a in ref["answers"].items()} != \
                {q: decision(a) for q, a in response["answers"].items()}:
            mismatches.append(case["name"])
    record("async_judge_vs_judge_fp32_cpu", {"cases": len(cases), "calls": calls,
                                            "max_margin_drift": drift, "decision_mismatches": mismatches})
    assert max(calls) > 1 and sum(calls) == len(cases)
    assert not mismatches and drift < 1e-3


# ----------------------------------------------------------------- VllmJudge
class StandInEngine:
    """`AsyncLLM.encode` / `abort` / `shutdown`, scoring each row with `kernel` (no cache)."""

    def __init__(self, enc, delay: float = 0.0):
        self.enc, self.delay = enc, delay
        self.submitted: list[str] = []
        self.aborted: list[str] = []

    async def encode(self, prompt, params, request_id):
        import torch

        from frida_decisions.vllm_backend import kernel
        from frida_decisions.vllm_backend.rows import parse_row

        self.submitted.append(request_id)
        if self.delay:
            await asyncio.sleep(self.delay)
        ids = prompt["prompt_token_ids"]
        parsed = parse_row(ids)
        with torch.inference_mode():
            hidden = self.enc.forward_rows(torch.tensor(ids), [(0, kernel.RowTensors(parsed, "cpu"), None)])
            margins = self.enc.readout(hidden, parsed.options)
        yield SimpleNamespace(prompt_token_ids=list(ids), num_cached_tokens=0,
                              outputs=SimpleNamespace(data=margins))

    async def abort(self, request_ids):
        self.aborted.extend(request_ids)

    def shutdown(self):
        pass


@pytest.fixture(scope="module")
def tiny(model_dir):
    """`Judge` around a small random T5, and the backend's encoder over the same weights."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_vllm_backend import make_tiny
    return make_tiny(model_dir)


def _vllm_judge(model_dir, enc, delay=0.0, max_model_len=2048):
    from test_vllm_backend import vocab

    from frida_decisions.base import BaseJudge
    from frida_decisions.vllm_backend.engine import VllmJudge

    return VllmJudge(StandInEngine(enc, delay), model_dir, None, vocab(BaseJudge(model_dir)),
                     max_model_len, state_max=384)


def test_vllm_judge_matches_judge(tiny, cases, model_dir):
    import torch

    judge, enc = tiny
    vj = _vllm_judge(model_dir, enc)

    async def main():
        return await asyncio.gather(*(vj.judge(c["request"]) for c in cases))

    out = asyncio.run(main())
    worst = 0.0
    for case, response in zip(cases, out):
        ref = torch.tensor(judge.margins_packed(case["request"]))
        ours = torch.tensor(flat_margins(response))
        worst = max(worst, float((ref - ours).abs().max()))
        assert response["usage"]["backend"] == "vllm" and response["usage"]["rows"] >= 1
    assert len(vj.engine.submitted) == len(set(vj.engine.submitted))       # one id per row
    assert worst <= 2e-5, worst


def test_vllm_judge_cancel_aborts_rows(tiny, model_dir):
    _, enc = tiny
    vj = _vllm_judge(model_dir, enc, delay=1.0)
    request = next(c for c in load_cases() if c["name"] == "generated/intent-catalog-40")["request"]

    async def main():
        task = asyncio.create_task(vj.judge(request))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert len(vj.engine.submitted) > 1 and sorted(vj.engine.aborted) == sorted(vj.engine.submitted)


def test_vllm_judge_refuses_rows_over_the_engine_limit(tiny, model_dir):
    _, enc = tiny
    vj = _vllm_judge(model_dir, enc, max_model_len=64)
    with pytest.raises(RequestError):
        asyncio.run(vj.judge(next(c for c in load_cases() if c["name"] == "generated/long-state")["request"]))
    assert not vj.engine.submitted
