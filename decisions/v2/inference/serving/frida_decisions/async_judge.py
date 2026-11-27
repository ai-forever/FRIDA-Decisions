"""`AsyncJudge`: an async front end for `Judge` (PyTorch), `OnnxJudge` or `MlxJudge`.

    judge = AsyncJudge.from_pretrained("ai-forever/FRIDA-Decisions")    # torch, GPU if available
    response = await judge.judge(request)                                # the same response as Judge
    responses = await asyncio.gather(*(judge.judge(r) for r in requests))
    await judge.close()

The model runs in one worker thread, so the event loop stays free while it
computes and the model is only ever called from that thread. Requests awaited
at the same time are scored together: while one batch runs, the next ones
queue up, and the worker hands them to `judge_batch` in one call (up to
`max_batch` requests), whose packed rows share encoder forwards.

Works wherever the wrapped judge works, CPU or GPU, any OS. For many
concurrent users on a GPU, `VllmJudge` (`frida-decisions[vllm]`, Linux) adds
continuous batching and a state cache shared across requests.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .constants import DEFAULT_REPO_ID
from .protocol import Calibration, parse_request


class AsyncJudge:
    """Async wrapper over a synchronous judge (anything with `judge_batch`)."""

    def __init__(self, judge, max_batch: int = 8, max_wait_ms: float = 0.0):
        """max_batch: requests scored in one `judge_batch` call.
        max_wait_ms: how long the worker waits for more requests after the first
            one of a batch arrives; 0 takes what is queued at that moment (requests
            arriving while a batch runs still form the next batch)."""
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.wrapped = judge
        self.max_batch = int(max_batch)
        self.max_wait = float(max_wait_ms) / 1000.0
        self.backend = f"async-{getattr(judge, 'backend', 'judge')}"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frida-decisions")
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inflight: list = []          # the batch the worker thread is scoring now
        self._closed = False

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = DEFAULT_REPO_ID, *, backend: str = "torch",
                        max_batch: int = 8, max_wait_ms: float = 0.0, **kwargs) -> "AsyncJudge":
        """Load `Judge` (backend="torch"), `OnnxJudge` (backend="onnx") or `MlxJudge`
        (backend="mlx") and wrap it.

        kwargs go to that backend's `from_pretrained` (device, dtype, state_max,
        threads, ...). For torch, `rows_per_forward` defaults to 16 here, so that a
        batch of large requests does not build one huge forward.
        """
        if backend == "torch":
            from .torch_backend import Judge
            kwargs.setdefault("rows_per_forward", 16)
            judge = Judge.from_pretrained(path_or_repo, **kwargs)
        elif backend == "onnx":
            from .onnx_backend import OnnxJudge
            judge = OnnxJudge.from_pretrained(path_or_repo, **kwargs)
        elif backend == "mlx":
            from .mlx_backend import MlxJudge
            judge = MlxJudge.from_pretrained(path_or_repo, **kwargs)
        else:
            raise ValueError(f"backend must be 'torch', 'onnx' or 'mlx', not {backend!r}")
        return cls(judge, max_batch, max_wait_ms)

    # ------------------------------------------------------------------ requests
    async def judge(self, request: dict, calibration: Calibration | None = None) -> dict:
        """Answer every question of one request. Raises `RequestError` for an invalid one."""
        parse_request(request)                 # invalid requests fail here, alone
        future = asyncio.get_running_loop().create_future()
        self._ensure_worker().put_nowait((request, calibration, future))
        return await future

    async def judge_batch(self, requests: list[dict], calibration: Calibration | None = None) -> list[dict]:
        return list(await asyncio.gather(*(self.judge(r, calibration) for r in requests)))

    async def __call__(self, request: dict, calibration: Calibration | None = None) -> dict:
        return await self.judge(request, calibration)

    # ------------------------------------------------------------------ worker
    def _ensure_worker(self) -> asyncio.Queue:
        if self._closed:
            raise RuntimeError("AsyncJudge is closed")
        loop = asyncio.get_running_loop()
        if self._worker is None:
            self._loop, self._queue = loop, asyncio.Queue()
            self._worker = loop.create_task(self._work())
        elif loop is not self._loop:
            raise RuntimeError("AsyncJudge is bound to the event loop it was first used in")
        return self._queue

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                if not self._queue.empty():
                    batch.append(self._queue.get_nowait())
                    continue
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            batch = [item for item in batch if not item[2].done()]    # skip cancelled callers
            # One judge_batch call per calibration (a call takes one calibration).
            groups: dict = {}
            for item in batch:
                groups.setdefault(item[1], []).append(item)
            for calibration, items in groups.items():
                self._inflight = items
                try:
                    responses = await loop.run_in_executor(
                        self._executor, self.wrapped.judge_batch, [r for r, _, _ in items], calibration)
                except Exception as error:      # the model failed: every caller in the call learns it
                    for _, _, future in items:
                        if not future.done():
                            future.set_exception(error)
                    continue
                for (_, _, future), response in zip(items, responses):
                    if not future.done():
                        future.set_result(response)
            self._inflight = []

    # ------------------------------------------------------------------ lifecycle
    async def close(self) -> None:
        """Stop the worker; requests queued or being scored fail with RuntimeError."""
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            pending = list(self._inflight)
            while not self._queue.empty():
                pending.append(self._queue.get_nowait())
            for _, _, future in pending:
                if not future.done():
                    future.set_exception(RuntimeError("AsyncJudge was closed"))
        self._executor.shutdown(wait=True)

    async def __aenter__(self) -> "AsyncJudge":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
