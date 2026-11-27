"""ONNX Runtime backend: int8 CPU inference without torch.

The graph takes the packed layout as inputs (`input_ids`, `buckets`,
`allowed`) and returns one head score per token; a candidate's margin is the
mean over its own tokens. Packing is the same numpy code the torch backend
uses. There is no state cache here: every row carries its state.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import BaseJudge, resolve_model_dir
from .constants import DEFAULT_REPO_ID, ONNX_INT8_FILE, ONNX_SUBDIR
from .packing import TokenizedRequest, build_rows, mean_readout, pack_rows

_ONNX_FILES = ["*.json", f"{ONNX_SUBDIR}/*"]


class OnnxJudge(BaseJudge):
    """Same API as `Judge`, backed by onnxruntime.

        judge = OnnxJudge.from_pretrained("ai-forever/FRIDA-Decisions", threads=8)
        judge({"state": "...", "questions": {...}})["answers"]
    """

    backend = "onnx-int8"

    def __init__(self, folder: Path, session, state_max: int | None = None,
                 rows_per_forward: int | None = 4):
        super().__init__(folder, state_max)
        self.session = session
        self.rows_per_forward = rows_per_forward

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = DEFAULT_REPO_ID, state_max: int = 384,
                        threads: int | None = None, providers: list | None = None,
                        rows_per_forward: int | None = 4, onnx_file: str = ONNX_INT8_FILE,
                        revision: str | None = None) -> "OnnxJudge":
        """Load the int8 ONNX graph from an exported folder or Hugging Face repo.

        threads: intra-op threads (default: onnxruntime's choice).
        rows_per_forward: packed rows per session call; bounds the memory of
            the `(rows, heads, L, L)` attention bias. None runs all rows at once.
        """
        import onnxruntime as ort

        folder = resolve_model_dir(path_or_repo, _ONNX_FILES, revision)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = int(threads)
        path = os.fspath(Path(folder) / ONNX_SUBDIR / onnx_file)
        session = ort.InferenceSession(path, options,
                                       providers=providers or ["CPUExecutionProvider"])
        return cls(folder, session, state_max, rows_per_forward)

    def _score(self, requests: list[TokenizedRequest]):
        rows, per_request = build_rows(requests, self.config)
        step = self.rows_per_forward or len(rows)
        margins: list[float] = []
        tokens = 0
        for i in range(0, len(rows), step):
            batch = pack_rows(rows[i:i + step], self.config)
            scores = self.session.run(["token_scores"], {
                "input_ids": batch.input_ids,
                "buckets": batch.buckets,
                "allowed": batch.allowed,
            })[0]
            margins += mean_readout(scores, batch.readout).tolist()
            tokens += sum(batch.lengths)
        out, start = [], 0
        for n in per_request:
            out.append(margins[start:start + n])
            start += n
        return out, {"rows": len(rows), "encoder_tokens": tokens, "state_cache": "off"}
