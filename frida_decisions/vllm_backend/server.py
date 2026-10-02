"""IO processor: a request in, the `Judge` response out, over `POST /pooling`.

    POST /pooling
    {"model": "...", "data": {"state": "...", "questions": {...}}}
    -> {"request_id": ..., "created_at": ..., "data": {"model", "answers", "margins", "usage"}}

`data` is the request exactly as `Judge` takes it. `pre_process` validates and
compiles it with the package's own `parse_request` / `compile_request`,
tokenizes it and lays it out into rows with `packing.layout_rows`
(`rows.tokenize_rows`, `rows.build_rows`), and hands vLLM one prompt per row.
`post_process` puts each row's margins back in candidate order and runs
`aggregate` -- the function every backend answers with.

Packing limits and instruction texts come from the model's
`decisions_config.json`. The state is cut at 384 tokens, as in
`Judge.from_pretrained`; `FRIDA_DECISIONS_STATE_MAX` overrides it.

Pooling task: `classify`, with `skip_reading_prefix_cache=False` set explicitly:
with the `plugin` task vLLM would silently turn the cache off for these requests.
"""
from __future__ import annotations

import os
import threading
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass

from vllm.exceptions import VLLMValidationError
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

from ..base import resolve_model_dir
from ..config import DecisionsConfig
from ..constants import PRODUCT_NAME
from ..packing import TextEncoder
from ..protocol import RequestError, aggregate, compile_request, parse_request
from .rows import LayoutError, build_rows, tokenize_rows

DEFAULT_STATE_MAX = 384            # `Judge.from_pretrained`'s default

# A request whose engine work fails or is cancelled never reaches post_process,
# and vLLM gives the IO processor no abort hook: its entry is dropped after
# PENDING_TTL seconds, and the table never holds more than MAX_IN_FLIGHT.
MAX_IN_FLIGHT = 1024
PENDING_TTL = 600.0


@dataclass
class Pending:
    request: object
    candidates: list
    row_candidates: list[list[int]]
    row_ids: list[array]           # what was sent, to see whether vLLM changed it
    state_tokens: int
    state_truncated: bool
    special_split: list[str]
    born: float


class DecisionsIO(IOProcessor):
    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        mc = vllm_config.model_config
        folder = resolve_model_dir(mc.model, ["*.json"], mc.revision)
        self.config = DecisionsConfig.load(folder)
        self.text = TextEncoder(folder / "tokenizer.json")
        self.state_max = int(os.environ.get("FRIDA_DECISIONS_STATE_MAX", DEFAULT_STATE_MAX))
        self.vocab = mc.hf_config.vocab_size
        self.max_model_len = mc.max_model_len
        self._pending: OrderedDict[str, Pending] = OrderedDict()
        self._lock = threading.Lock()

    def parse_data(self, data: object):
        try:
            return parse_request(data)
        except RequestError as error:
            raise VLLMValidationError(error.message, parameter=error.field) from None

    def merge_pooling_params(self, params: PoolingParams | None = None) -> PoolingParams:
        return PoolingParams(task="classify", skip_reading_prefix_cache=False)

    def pre_process(self, prompt, request_id: str | None = None, **kwargs):
        cfg = self.config
        try:
            candidates = compile_request(prompt, cfg.instruction_suffixes, cfg.yes_no_default_criteria)
            tok, split = tokenize_rows(candidates, self.text, cfg, self.state_max)
            rows = build_rows(tok, cfg, self.vocab)
        except RequestError as error:
            raise VLLMValidationError(error.message, parameter=error.field) from None
        except LayoutError as error:
            raise VLLMValidationError(str(error)) from None
        longest = max(len(r.ids) for r in rows)
        if longest > self.max_model_len:
            raise VLLMValidationError(
                f"a row of this request is {longest} tokens, over --max-model-len "
                f"{self.max_model_len}; raise it or lower FRIDA_DECISIONS_STATE_MAX")
        now = time.monotonic()
        with self._lock:
            if request_id in self._pending:
                raise VLLMValidationError(f"request id {request_id!r} is already in flight")
            while self._pending and (len(self._pending) >= MAX_IN_FLIGHT or
                                     now - next(iter(self._pending.values())).born > PENDING_TTL):
                self._pending.popitem(last=False)
            self._pending[request_id] = Pending(
                prompt, candidates, [r.candidates for r in rows],
                [array("i", r.ids) for r in rows], len(tok.state),
                self.text.count(candidates[0].state) > len(tok.state), split, now)
        return [{"prompt_token_ids": row.ids} for row in rows]

    def post_process(self, model_output, request_id: str | None = None, **kwargs):
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            raise RuntimeError(f"request {request_id!r}: its rows were dropped (over "
                               f"{MAX_IN_FLIGHT} in flight, or older than {PENDING_TTL:.0f} s)")
        request, candidates = pending.request, pending.candidates
        if len(model_output) != len(pending.row_candidates):
            raise RuntimeError(f"{len(pending.row_candidates)} rows sent, "
                               f"{len(model_output)} came back")
        margins = [float("nan")] * len(candidates)
        cached = computed = 0
        for row_candidates, sent, out in zip(pending.row_candidates, pending.row_ids, model_output):
            if array("i", out.prompt_token_ids) != sent:
                raise VLLMValidationError(
                    "vLLM changed the token ids of a row (truncate_prompt_tokens, padding or "
                    "truncation_side in the request?); rows must reach the model intact")
            values = out.outputs.data.float().tolist()
            if len(values) != len(row_candidates) or any(v != v for v in values):
                raise RuntimeError(f"row of {len(row_candidates)} options returned {len(values)} "
                                   "values: the model rejected the row (see the server log)")
            for index, value in zip(row_candidates, values):
                margins[index] = value
            cached += out.num_cached_tokens
            computed += len(out.prompt_token_ids) - out.num_cached_tokens
        answers = aggregate(request, candidates, margins)
        by_question: dict[str, dict[str, float]] = {}
        for cand, margin in zip(candidates, margins):
            by_question.setdefault(cand.question_id, {})[cand.option_id] = margin
        return {"model": PRODUCT_NAME,
                "answers": answers,
                "margins": by_question,
                "usage": {"state_tokens": pending.state_tokens,
                          "state_truncated": pending.state_truncated,
                          "questions": len(request.questions),
                          "candidates": len(candidates),
                          "rows": len(pending.row_candidates),
                          "prompt_tokens": cached + computed,
                          "cached_tokens": cached,
                          "special_tokens_split": pending.special_split,
                          "backend": "vllm"}}
