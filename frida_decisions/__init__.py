"""Structured decisions over text with one encoder pass: choice, score, yes/no, ranking."""
from __future__ import annotations

from .config import DecisionsConfig
from .constants import DEFAULT_REPO_ID, PRODUCT_NAME, QuestionType
from .protocol import Calibration, RequestError, aggregate, compile_request, decision, parse_request

__version__ = "0.1.1"

__all__ = [
    "PRODUCT_NAME", "DEFAULT_REPO_ID", "QuestionType", "DecisionsConfig",
    "Calibration", "RequestError", "parse_request", "compile_request", "aggregate", "decision",
    "Judge", "OnnxJudge",
]


def __getattr__(name):
    # Backends are imported lazily: each one needs only its own extra.
    if name == "Judge":
        try:
            from .torch_backend import Judge
        except ImportError as error:
            raise ImportError("Judge needs PyTorch: pip install 'frida-decisions[torch]'") from error
        return Judge
    if name == "OnnxJudge":
        try:
            import onnxruntime  # noqa: F401
        except ImportError as error:
            raise ImportError("OnnxJudge needs onnxruntime: pip install 'frida-decisions[onnx]'") from error
        from .onnx_backend import OnnxJudge
        return OnnxJudge
    raise AttributeError(name)
