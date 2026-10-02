"""Structured decisions over text with one encoder pass: choice, score, yes/no, ranking."""
from __future__ import annotations

from .config import DecisionsConfig
from .constants import DEFAULT_REPO_ID, PRODUCT_NAME, QuestionType
from .protocol import Calibration, RequestError, aggregate, compile_request, decision, parse_request

__version__ = "0.1.0"

__all__ = [
    "PRODUCT_NAME", "DEFAULT_REPO_ID", "QuestionType", "DecisionsConfig",
    "Calibration", "RequestError", "parse_request", "compile_request", "aggregate", "decision",
    "Judge", "OnnxJudge",
]


def __getattr__(name):
    # Backends are imported lazily: the ONNX path must work without torch.
    if name == "Judge":
        from .torch_backend import Judge
        return Judge
    if name == "OnnxJudge":
        from .onnx_backend import OnnxJudge
        return OnnxJudge
    raise AttributeError(name)
