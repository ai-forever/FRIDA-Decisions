"""A vLLM pooling model that scores a complete question tree with the FRIDA flat executor."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Set
from pathlib import Path

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.models.interfaces import IsAttentionFree
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.pool.metadata import PoolingMetadata

from frida_decisions.v2.model.vllm_hybrid import SPEC, VISION
from frida_decisions.v2.model.vllm_hybrid.media import from_params
from frida_decisions.v2.model.vllm_hybrid.rows import RowError, decode

logger = init_logger(f"vllm.plugins.{__name__}")    # vLLM configures only "vllm.*"

SEED = 0
TRACE = os.environ.get("FDD_HYBRID_TRACE")          # path: one JSON line per step, with a GPU sync




def with_conv(encoder):
    from frida_decisions.v2.model.vllm_hybrid.kernels import install_conv

    cell = next(layer.linear_attn for layer in encoder.backbone.layers
                if layer.block_type == "linear_attention")
    logger.info("FridaHybridModel: convolution by %s",
                install_conv(cell, os.environ.get("FDD_HYBRID_CONV", "fla") == "fla"))
    return encoder


class TowerClock:
    """Time in the vision tower, with a sync on each side: only under FDD_HYBRID_TRACE."""

    def __init__(self, vision):
        self.seconds, self.images, self.features = 0.0, 0, vision.features
        vision.features = self

    def __call__(self, mm):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = self.features(mm)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - started
        self.images += int(mm["image_grid_thw"].shape[0])
        return out

    def take(self) -> tuple[float, int]:
        out = self.seconds, self.images
        self.seconds, self.images = 0.0, 0
        return out


class HybridPooler(Pooler):
    """Owns the encoder (see the module doc), so that its parameters are the model's."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder: nn.Module | None = None
        self.executor = None
        self.nan = None
        self.tower = None                              # `TowerClock` under a trace

    def get_supported_tasks(self) -> Set[PoolingTask]:
        # vLLM's default model runner serves no "plugin" task; the IO processor sends classify.
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    def forward(self, hidden_states: torch.Tensor,
                pooling_metadata: PoolingMetadata) -> list[torch.Tensor]:
        started = time.perf_counter()
        ids = hidden_states.tolist()
        plans, read, start = [], [], 0
        for length, params in zip(pooling_metadata.prompt_lens.tolist(), pooling_metadata.pooling_params):
            try:
                plans.append(decode(ids[start:start + length], from_params(params.extra_kwargs, defer_rgb8=True)))
                read.append(True)
            except (RowError, KeyError, ValueError, RuntimeError):
                read.append(False)
            start += length
        if not plans:
            return [self.nan for _ in read]
        device = hidden_states.device
        decoded = time.perf_counter()
        batch = self.executor.pack(plans).to(device)
        packed = time.perf_counter()
        margins = self.executor.forward(batch).float()
        if TRACE:
            torch.cuda.synchronize()
            done = time.perf_counter()
            tower = self.tower.take() if self.tower else (0.0, 0)
            with open(TRACE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": done, "requests": len(plans), "tokens": len(ids),
                                    "decode_ms": 1e3 * (decoded - started),
                                    "pack_ms": 1e3 * (packed - decoded),
                                    "forward_ms": 1e3 * (done - packed),
                                    "tower_ms": 1e3 * tower[0], "images": tower[1]}) + "\n")
        taken = iter(margins.split([plan.n_slots for plan in plans]))
        return [next(taken) if ok else self.nan for ok in read]


@attn_type("attention_free")
@default_pooling_type(seq_pooling_type="LAST")      # a label for vLLM's config; the pooler is ours
class FridaHybridModel(nn.Module, IsAttentionFree):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self._check_config(vllm_config)
        self.dtype = vllm_config.model_config.dtype
        self.folder = Path(vllm_config.model_config.model)
        self.device_name = vllm_config.device_config.device
        self.pooler = HybridPooler()

    @staticmethod
    def _check_config(vllm_config: VllmConfig) -> None:
        sched, par = vllm_config.scheduler_config, vllm_config.parallel_config
        problems = []
        if sched.enable_chunked_prefill:
            problems.append("chunked prefill is on: pass --no-enable-chunked-prefill")
        if sched.long_prefill_token_threshold:
            problems.append("--long-prefill-token-threshold must be 0")
        if par.tensor_parallel_size != 1 or par.pipeline_parallel_size != 1:
            problems.append("tensor and pipeline parallelism are not implemented")
        if problems:
            raise ValueError("FridaHybridModel: " + "; ".join(problems))
        if not vllm_config.model_config.enforce_eager:
            logger.warning("FridaHybridModel is meant to run with --enforce-eager")

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return input_ids

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors=None, inputs_embeds: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        """vLLM's model runner V2 passes `embed_input_ids`' output, here the ids, in place of them."""
        return input_ids if inputs_embeds is None else inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """`weights` is left unread: the serving folder's own (`weights/`, if it has them) are
        read by `weights.load`. Returning None turns off vLLM's check that every parameter
        came from it."""
        from importlib.util import find_spec

        spec = json.loads((self.folder / SPEC).read_text(encoding="utf-8"))
        cap = os.environ.get("FDD_HYBRID_MEMORY_GIB")  # a smaller card, simulated: the allocator's limit
        if cap:
            total = torch.cuda.get_device_properties(self.device_name).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * (1 << 30) / total))
            logger.info("FridaHybridModel: PyTorch limited to %s GiB of %.1f", cap, total / (1 << 30))
        with set_default_torch_dtype(torch.float32):   # as a PyTorch run has it
            started = time.perf_counter()
            if "weights" in spec:
                from frida_decisions.v2.model.vllm_hybrid.weights import load

                encoder = with_conv(load(self.folder, self.device_name, self.dtype))
            else:
                raise ValueError("A standalone FRIDA Decisions checkpoint with weights is required")
            self.pooler.encoder = encoder.to(self.device_name).eval()
            logger.info("FridaHybridModel: %s in %.1f s, %d MiB on the card, peak %d MiB",
                        "weights read from the folder" if "weights" in spec else "loaded and merged",
                        time.perf_counter() - started, torch.cuda.memory_allocated() >> 20,
                        torch.cuda.max_memory_allocated() >> 20)
            self.pooler.executor = self.pooler.encoder.executor
            self.pooler.nan = torch.full((1,), float("nan"), device=self.device_name)
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            getattr(self.pooler.executor, "quantize_experts", lambda: None)()   # flat_fp8: experts in FP8
            empty = os.environ.get("FDD_HYBRID_WARMUP_EMPTY") == "1"

            def after_pass(i: int) -> None:
                logger.info("FridaHybridModel: warm-up pass %d: peak %d MiB allocated, %d MiB reserved", i,
                            torch.cuda.max_memory_allocated() >> 20, torch.cuda.max_memory_reserved() >> 20)
                if empty:
                    torch.cuda.empty_cache()

            encoder.warm_up(after_pass)                # the kernels, before the first request
        logger.info("FridaHybridModel: warm-up %.1f s, peak %d MiB allocated, %d MiB reserved",
                    time.perf_counter() - started, torch.cuda.max_memory_allocated() >> 20,
                    torch.cuda.max_memory_reserved() >> 20)
        if TRACE and encoder.vision is not None:
            self.pooler.tower = TowerClock(encoder.vision)
        logger.info("FridaHybridModel: %s, %s, %d layers, %s, fla %s, causal_conv1d %s",
                    spec["variant"], self.dtype, len(encoder.backbone.layers),
                    type(self.pooler.executor).__name__, find_spec("fla") is not None,
                    find_spec("causal_conv1d") is not None)
