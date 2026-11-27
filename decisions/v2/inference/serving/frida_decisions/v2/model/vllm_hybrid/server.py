"""IO processor: a request in, its answers out, over `POST /pooling`.

`data` is the request as `protocol.parse_request` takes it, plus `images`: a list of
base64 images (bare or as data URLs) for the state, for a model served with its vision
tower. `pre_process` renders the tree in the API server's process -- images decoded,
the checkpoint's image processor, the state's M-RoPE rows -- and hands vLLM one prompt
(`rows.encode`), the media beside it in the pooling params (`media`). `post_process`
takes one margin per option back and runs `protocol.aggregate`. vLLM's task is
`classify`, as in fd's backend: its model runner serves no `plugin` task.

vLLM calls `merge_pooling_params` right after `pre_process` for the same request, with
no request id, and on the same thread; the media wait for it in `_media`.
"""
from __future__ import annotations

import os

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from vllm.exceptions import VLLMValidationError
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

from frida_decisions.v2.model.render.causal_scalar import CausalScalarConfig, CausalScalarRenderer
from frida_decisions.v2.model.vllm_hybrid import SPEC
from frida_decisions.v2.model.vllm_hybrid.media import decode_image, to_params, vision_pixel_dtype
from frida_decisions.v2.model.vllm_hybrid.rows import RowError, encode
from frida_decisions.v2.protocol import ProtocolError, Request, aggregate, compile_request, parse_request

# A request whose engine work fails never reaches `post_process`, and vLLM gives an IO
# processor no abort hook: its entry is dropped after PENDING_TTL seconds.
MAX_IN_FLIGHT = 4096
PENDING_TTL = 600.0
MAX_IMAGES = 16


@dataclass
class Pending:
    request: Request
    tasks: list
    tokens: int
    born: float
    render_ms: float
    images: int


def vision_inputs(base: str, max_pixels: int | None):
    """The renderer's `media` without the model's weights: the processor, and the VL model
    built on the meta device, whose `get_rope_index` reads only its config."""
    import torch
    from transformers import AutoModel, AutoProcessor

    from frida_decisions.v2.model.causal import load_config
    from frida_decisions.v2.model.vision import VisionInputs

    with torch.device("meta"):
        vl = AutoModel.from_config(load_config(base))
    return VisionInputs(vl, AutoProcessor.from_pretrained(base).image_processor, max_pixels=max_pixels)


class HybridIO(IOProcessor):
    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        from transformers import Qwen2Tokenizer

        folder = Path(vllm_config.model_config.model)
        spec = json.loads((folder / SPEC).read_text(encoding="utf-8"))
        # This is the root tokenizer used by the released renderer. AutoTokenizer
        # probes the custom model config and prompts on an interactive terminal.
        tokenizer = Qwen2Tokenizer.from_pretrained(folder, fix_mistral_regex=False)
        self.renderer = CausalScalarRenderer(tokenizer, CausalScalarConfig.from_dict(spec["causal_scalar"]))
        if "vision" in spec:
            base = str(folder / "base")
            self.renderer.media = vision_inputs(base, spec["vision"].get("max_pixels"))
        self.pixel_dtype = vision_pixel_dtype(folder, spec) if os.getenv("FDD_FLAT_TEXT_PACK") == "parallel" else None
        if self.pixel_dtype is not None and self.renderer.media is not None:
            from frida_decisions.v2.model.vllm_hybrid.media import rgb8_normalization_table
            self.renderer.media.wire_rgb8_table = rgb8_normalization_table(self.renderer.media.image_processor)
        if os.getenv("FDD_FLAT_TEXT_PACK") == "parallel":
            self.renderer.enable_serving_tokenization()
        self.max_model_len = vllm_config.model_config.max_model_len
        self.batch_suffix = os.getenv("FDD_FLAT_TEXT_PACK") in {"fast", "parallel"}
        self._pending: OrderedDict[str, Pending] = OrderedDict()
        self._lock = threading.Lock()
        self._media = threading.local()

    def parse_data(self, data: object) -> tuple[Request, list]:
        images = []
        if isinstance(data, dict) and "images" in data:
            data = dict(data)
            images = data.pop("images")
            if not isinstance(images, list) or not all(isinstance(i, str) for i in images):
                raise VLLMValidationError("images must be a list of base64 strings", parameter="images")
            if len(images) > MAX_IMAGES:
                raise VLLMValidationError(f"at most {MAX_IMAGES} images", parameter="images")
            if images and self.renderer.media is None:
                raise VLLMValidationError("this model is served without its vision tower", parameter="images")
        try:
            return parse_request(data), images
        except ProtocolError as error:
            raise VLLMValidationError(str(error)) from None

    def merge_pooling_params(self, params: PoolingParams | None = None) -> PoolingParams:
        extra = getattr(self._media, "next", None)
        self._media.next = None
        return PoolingParams(task="classify", use_activation=False, extra_kwargs=extra)

    def pre_process(self, prompt, request_id: str | None = None, **kwargs):
        request, images = prompt
        started = time.perf_counter()
        try:
            decoded = [decode_image(i) for i in images]
            tasks = compile_request(request)
            plan = self.renderer.render(tasks, decoded or None, batch_suffix=self.batch_suffix)
            ids = encode(plan)
        except (ProtocolError, RowError, ValueError, OSError) as error:
            raise VLLMValidationError(str(error)) from None
        if len(ids) > self.max_model_len:
            raise VLLMValidationError(f"this request is {len(ids)} tokens, over the server's "
                                      f"limit of {self.max_model_len}: raise --max-model-len")
        self._media.next = to_params(plan, self.pixel_dtype) if plan.mm else None
        now = time.monotonic()
        with self._lock:
            while self._pending and (len(self._pending) >= MAX_IN_FLIGHT or
                                     now - next(iter(self._pending.values())).born > PENDING_TTL):
                self._pending.popitem(last=False)
            self._pending[request_id] = Pending(request, tasks, len(ids), now,
                                                1e3 * (time.perf_counter() - started), len(images))
        return {"prompt_token_ids": ids}

    def post_process(self, model_output, request_id: str | None = None, **kwargs) -> dict:
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            raise RuntimeError(f"request {request_id!r}: dropped (over {MAX_IN_FLIGHT} in flight, "
                               f"or older than {PENDING_TTL:.0f} s)")
        if len(model_output) != 1:
            raise RuntimeError(f"one prompt sent, {len(model_output)} came back")
        margins = model_output[0].outputs.data.float().tolist()
        if len(margins) != len(pending.tasks):
            raise RuntimeError(f"{len(pending.tasks)} options sent, {len(margins)} margins came "
                               "back: the model could not read the prompt")
        by_question: dict[str, dict[str, float]] = {}
        for task, margin in zip(pending.tasks, margins):
            by_question.setdefault(task.question_id, {})[task.option_id] = margin
        return {"answers": aggregate(pending.request, pending.tasks, margins),
                "margins": by_question,
                "usage": {"prompt_tokens": pending.tokens, "options": len(margins), "images": pending.images,
                          "render_ms": round(pending.render_ms, 3), "backend": "vllm_hybrid"}}
