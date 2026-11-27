"""IO processor: a request in, the `Judge` response out, over `POST /pooling`.

    POST /pooling
    {"model": "...", "data": {"state": "...", "questions": {...}}}
    -> {"request_id": ..., "created_at": ..., "data": {"model", "answers", "margins", "usage"}}

`data` is the request exactly as `Judge` takes it. `pre_process` validates,
compiles and lays it out into rows (`prepare.prepare`: the package's own
`parse_request` / `compile_request`, rows by `packing.layout_rows`) and hands
vLLM one prompt per row. `post_process` puts each row's margins back in
candidate order and runs `aggregate` (`prepare.respond`), the function every
backend answers with. `VllmJudge` (`engine.py`) runs the same two steps around
an in-process engine.

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
from ..packing import TextEncoder
from ..protocol import RequestError, parse_request
from .prepare import Prepared, prepare, respond
from .rows import LayoutError

DEFAULT_STATE_MAX = 384            # `Judge.from_pretrained`'s default

# A request whose engine work fails or is cancelled never reaches post_process,
# and vLLM gives the IO processor no abort hook: its entry is dropped after
# PENDING_TTL seconds, and the table never holds more than MAX_IN_FLIGHT.
MAX_IN_FLIGHT = 1024
PENDING_TTL = 600.0


@dataclass
class Pending:
    prepared: Prepared
    row_ids: list[array]           # what was sent, to see whether vLLM changed it
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
        try:
            prepared = prepare(prompt, self.text, self.config, self.state_max, self.vocab,
                               self.max_model_len)
        except RequestError as error:
            raise VLLMValidationError(error.message, parameter=error.field) from None
        except LayoutError as error:
            raise VLLMValidationError(str(error)) from None
        now = time.monotonic()
        with self._lock:
            if request_id in self._pending:
                raise VLLMValidationError(f"request id {request_id!r} is already in flight")
            while self._pending and (len(self._pending) >= MAX_IN_FLIGHT or
                                     now - next(iter(self._pending.values())).born > PENDING_TTL):
                self._pending.popitem(last=False)
            self._pending[request_id] = Pending(prepared, [array("i", r.ids) for r in prepared.rows],
                                                now)
        return [{"prompt_token_ids": row.ids} for row in prepared.rows]

    def post_process(self, model_output, request_id: str | None = None, **kwargs):
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            raise RuntimeError(f"request {request_id!r}: its rows were dropped (over "
                               f"{MAX_IN_FLIGHT} in flight, or older than {PENDING_TTL:.0f} s)")
        if len(model_output) != len(pending.row_ids):
            raise RuntimeError(f"{len(pending.row_ids)} rows sent, {len(model_output)} came back")
        row_margins, cached, prompt = [], 0, 0
        for sent, out in zip(pending.row_ids, model_output):
            if array("i", out.prompt_token_ids) != sent:
                raise VLLMValidationError(
                    "vLLM changed the token ids of a row (truncate_prompt_tokens, padding or "
                    "truncation_side in the request?); rows must reach the model intact")
            row_margins.append(out.outputs.data.float().tolist())
            cached += out.num_cached_tokens
            prompt += len(out.prompt_token_ids)
        return respond(pending.prepared, row_margins, cached, prompt)
