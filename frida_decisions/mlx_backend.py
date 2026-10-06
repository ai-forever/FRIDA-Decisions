"""Packed FRIDA inference with MLX on Apple Silicon."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import mlx.core as mx

from .base import BaseJudge, resolve_model_dir
from .constants import DEFAULT_REPO_ID
from .mlx_modeling import FridaMlxDecisionModel, StateCache
from .packing import (TokenizedRequest, build_rows, count_packed_rows, pack_queries,
                      pack_rows, state_buckets)

_MLX_FILES = ["*.json", "model.safetensors", "head.safetensors"]


class MlxJudge(BaseJudge):
    """The shared decision API with a native MLX encoder and FP32 head."""

    backend = "mlx"

    def __init__(self, folder: Path, encoder: FridaMlxDecisionModel,
                 state_max: int | None = None, rows_per_forward: int | None = 1,
                 state_cache_mb: int = 512, compile_encoder: bool = True):
        super().__init__(folder, state_max)
        if rows_per_forward is not None and (not isinstance(rows_per_forward, int)
                                             or rows_per_forward < 1):
            raise ValueError("rows_per_forward must be a positive integer or None")
        if state_cache_mb < 0:
            raise ValueError("state_cache_mb must be nonnegative")
        self.state_cache = StateCache(state_cache_mb * 2**20) if state_cache_mb else None
        self.model = encoder
        self.dtype = encoder.embed.weight.dtype
        self.rows_per_forward = rows_per_forward
        self.compile_encoder = compile_encoder
        self._forward_encoder = mx.compile(encoder.__call__) if compile_encoder else encoder.__call__
        self._forward_cached_encoder = (mx.compile(encoder.forward_cached) if compile_encoder
                                        else encoder.forward_cached)

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = DEFAULT_REPO_ID,
                        dtype=mx.float32, state_max: int = 384,
                        rows_per_forward: int | None = 1,
                        revision: str | None = None,
                        state_cache_mb: int = 512, compile_encoder: bool = True) -> "MlxJudge":
        """Load original released weights. FP32 is the default encoder precision.

        dtype accepts mx.float32 or mx.bfloat16. The head stays in FP32.
        rows_per_forward bounds the attention memory. None uses all rows.
        state_cache_mb bounds state tensor memory. Zero disables the cache.
        compile_encoder combines encoder operations. False disables compilation.
        revision pins a Hugging Face revision. Local folders also work.
        """
        folder = resolve_model_dir(path_or_repo, _MLX_FILES, revision)
        return cls(folder, FridaMlxDecisionModel.from_folder(folder, dtype),
                   state_max, rows_per_forward, state_cache_mb, compile_encoder)

    def use_state_cache(self, requests: list[TokenizedRequest]) -> bool:
        if self.state_cache is None or len(requests) != 1:
            return False
        req = requests[0]
        return tuple(req.state) in self.state_cache or count_packed_rows(req, self.config) > 1

    def _score(self, requests: list[TokenizedRequest]):
        if self.use_state_cache(requests):
            margins, usage = self._score_cached(requests[0])
            return [margins], usage
        return self._score_packed(requests)

    def _score_packed(self, requests: list[TokenizedRequest]):
        rows, per_request = build_rows(requests, self.config)
        if not rows:
            return [[] for _ in requests], {"rows": 0, "encoder_tokens": 0, "state_cache": "off"}
        step = self.rows_per_forward or len(rows)
        margins, tokens = [], 0
        for i in range(0, len(rows), step):
            batch = pack_rows(rows[i:i + step], self.config)
            margins += self._forward_packed(batch)
            tokens += sum(batch.lengths)
        out, start = [], 0
        for n in per_request:
            out.append(margins[start:start + n])
            start += n
        return out, {"rows": len(rows), "encoder_tokens": tokens, "state_cache": "off"}

    def _forward_packed(self, batch) -> list[float]:
        hidden = self._forward_encoder(mx.array(batch.input_ids, dtype=mx.int32),
                            mx.array(batch.buckets, dtype=mx.int32), mx.array(batch.allowed))
        r = batch.readout
        margins = self.model.margins(hidden, mx.array(r.row, dtype=mx.int32),
                                     mx.array(r.col, dtype=mx.int32),
                                     mx.array(r.slot, dtype=mx.int32), r.count)
        mx.eval(margins)
        return margins.tolist()

    def margins_packed(self, request: dict) -> list[float]:
        """Return margins in candidate order through the packed encoder."""
        return self._score_packed([self.compile(request)[2]])[0][0]

    def _score_cached(self, req: TokenizedRequest):
        if self.state_cache is None:
            raise ValueError("State cache is disabled")
        key = tuple(req.state)
        hit = self.state_cache.get(key)
        state_cost = 0
        if hit is None:
            n = len(req.state)
            buckets, allowed = state_buckets(n, self.config)
            ids = np.full((1, buckets.shape[0]), self.config.pad_token_id, dtype=np.int32)
            ids[0, :n] = req.state
            ks, vs = self.model.encode_state(mx.array(ids), mx.array(buckets[None]),
                                             mx.array(allowed[None]), n)
            self.state_cache.put(key, ks, vs)
            state_cost = n
        else:
            ks, vs = hit
        # Evaluate row chunks to preserve the attention memory limit.
        q = pack_queries(req, self.config)
        step = self.rows_per_forward or len(q.rows)
        margins = []
        for start in range(0, len(q.rows), step):
            end = start + step
            r = q.readout
            selected = (r.row >= start) & (r.row < end)
            slots = r.slot[selected]
            offset = int(slots[0])
            count = int(slots[-1]) - offset + 1
            hidden = self._forward_cached_encoder(mx.array(q.input_ids[start:end], dtype=mx.int32),
                mx.array(q.buckets[start:end]), mx.array(q.allowed[start:end]), ks, vs)
            values = self.model.margins(hidden, mx.array(r.row[selected] - start, dtype=mx.int32),
                mx.array(r.col[selected], dtype=mx.int32),
                mx.array(slots - offset, dtype=mx.int32), count)
            mx.eval(values)
            margins += values.tolist()
        return margins, {"rows": len(q.rows),
                         "encoder_tokens": state_cost + sum(len(row.ids) for row in q.rows),
                         "state_cache": "miss" if hit is None else "hit"}

    def margins_cached(self, request: dict) -> list[float]:
        """Return margins in candidate order through the cached encoder."""
        return self._score_cached(self.compile(request)[2])[0]
