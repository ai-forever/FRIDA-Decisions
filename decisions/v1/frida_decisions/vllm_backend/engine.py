"""`VllmJudge`: the vLLM engine inside your own process, with an async API.

    judge = VllmJudge.from_pretrained("ai-forever/FRIDA-Decisions", gpu_memory_utilization=0.45)
    response = await judge.judge(request)          # the same response as Judge
    responses = await asyncio.gather(*(judge.judge(r) for r in requests))
    judge.close()

The same engine `vllm serve` runs, without the HTTP server: requests awaited
concurrently are batched by vLLM's scheduler, and a text the engine has read is
reused through its prefix cache. A request is laid out into rows and its answer
assembled by `prepare.py`, the code the server's IO processor runs, so the
response is the server's -- and `Judge`'s.

The engine itself runs in a separate process that vLLM starts; this object only
submits rows and collects margins. Cancelling an awaiting `judge()` aborts its
rows in the engine.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..base import resolve_model_dir
from ..config import DecisionsConfig
from ..constants import DEFAULT_REPO_ID
from ..packing import TextEncoder
from ..protocol import Calibration
from .prepare import prepare, respond

DEFAULT_STATE_MAX = 384            # `Judge.from_pretrained`'s default


class VllmJudge:
    """Async judge over an in-process vLLM engine. Linux with a CUDA GPU."""

    backend = "vllm"

    def __init__(self, engine, folder: Path, pooling_params, vocab: int, max_model_len: int,
                 state_max: int = DEFAULT_STATE_MAX):
        self.engine = engine
        self.folder = Path(folder)
        self.config = DecisionsConfig.load(self.folder)
        self.text = TextEncoder(self.folder / "tokenizer.json")
        self.pooling_params = pooling_params
        self.vocab = vocab
        self.max_model_len = max_model_len
        self.state_max = int(state_max)

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = DEFAULT_REPO_ID, *, state_max: int = DEFAULT_STATE_MAX,
                        gpu_memory_utilization: float = 0.9, max_model_len: int = 2048,
                        enable_prefix_caching: bool = True, revision: str | None = None,
                        **engine_args) -> "VllmJudge":
        """Start the engine for an exported folder or a Hugging Face repo (blocks while loading).

        state_max: state tokens kept, as in `Judge` (384 by default; up to 512 is the
            trained range).
        gpu_memory_utilization: share of the GPU vLLM may take, weights and cache
            included; lower it on a card that also drives a display.
        enable_prefix_caching: reuse texts already read (the state cache).
        engine_args: anything else `vllm.AsyncEngineArgs` takes (`dtype`,
            `max_num_seqs`, ...). Chunked prefill and CUDA graphs are always off:
            the model's attention needs whole rows and runs eagerly.
        """
        from vllm import AsyncEngineArgs, PoolingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        from . import ARCH, register

        register()                     # also when the package is used from a source tree
        for fixed in ("enable_chunked_prefill", "enforce_eager", "hf_overrides", "runner"):
            if fixed in engine_args:
                raise ValueError(f"{fixed} is set by VllmJudge")
        folder = resolve_model_dir(path_or_repo, ["*.json"], revision)
        args = AsyncEngineArgs(
            model=str(path_or_repo), revision=revision, runner="pooling",
            hf_overrides={"architectures": [ARCH]},
            enable_chunked_prefill=False, enforce_eager=True,
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching, **engine_args)
        engine = AsyncLLM.from_engine_args(args)
        vocab = json.loads((folder / "config.json").read_text(encoding="utf-8"))["vocab_size"]
        params = PoolingParams(task="classify", skip_reading_prefix_cache=False)
        return cls(engine, folder, params, vocab, max_model_len, state_max)

    # ------------------------------------------------------------------ requests
    async def judge(self, request: dict, calibration: Calibration | None = None,
                    cache_salt: str | None = None) -> dict:
        """Answer every question of one request.

        cache_salt: texts are shared through the cache only between requests with
            the same salt (one per tenant, for example).
        Raises `RequestError` for an invalid request.
        """
        import asyncio

        started = time.perf_counter()
        prepared = prepare(request, self.text, self.config, self.state_max, self.vocab,
                           self.max_model_len)
        base = uuid.uuid4().hex
        ids = [f"fd-{base}-{i}" for i in range(len(prepared.rows))]

        async def run(row, request_id):
            prompt = {"prompt_token_ids": row.ids}
            if cache_salt is not None:
                prompt["cache_salt"] = cache_salt
            final = None
            async for out in self.engine.encode(prompt, self.pooling_params, request_id):
                final = out
            if final is None or list(final.prompt_token_ids) != row.ids:
                raise RuntimeError("the engine did not return the row it was given")
            return final

        try:
            outs = await asyncio.gather(*(run(r, i) for r, i in zip(prepared.rows, ids)))
        except BaseException:
            await self.engine.abort(ids)   # cancelled or failed: free the rest of the rows
            raise
        return respond(prepared, [o.outputs.data.float().tolist() for o in outs],
                       sum(o.num_cached_tokens for o in outs),
                       sum(len(o.prompt_token_ids) for o in outs), calibration,
                       {"milliseconds": round((time.perf_counter() - started) * 1000.0, 2)})

    async def judge_batch(self, requests: list[dict], calibration: Calibration | None = None) -> list[dict]:
        """Several requests at once; the engine batches their rows."""
        import asyncio

        return list(await asyncio.gather(*(self.judge(r, calibration) for r in requests)))

    async def __call__(self, request: dict, calibration: Calibration | None = None) -> dict:
        return await self.judge(request, calibration)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Stop the engine and its process."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    async def __aenter__(self) -> "VllmJudge":
        return self

    async def __aexit__(self, *exc) -> None:
        self.close()
