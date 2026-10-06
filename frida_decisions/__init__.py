"""Structured decisions over text with one encoder pass: choice, score, yes/no, ranking."""
from __future__ import annotations

from .config import DecisionsConfig
from .constants import DEFAULT_REPO_ID, PRODUCT_NAME, QuestionType
from .protocol import Calibration, RequestError, aggregate, compile_request, decision, parse_request

__version__ = "0.3.0"

__all__ = [
    "PRODUCT_NAME", "DEFAULT_REPO_ID", "QuestionType", "DecisionsConfig",
    "Calibration", "RequestError", "parse_request", "compile_request", "aggregate", "decision",
    "Judge", "OnnxJudge", "AsyncJudge", "VllmJudge", "MlxJudge",
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
    if name == "AsyncJudge":
        from .async_judge import AsyncJudge
        return AsyncJudge
    if name == "VllmJudge":
        try:
            import vllm  # noqa: F401
        except ImportError as error:
            raise ImportError("VllmJudge needs vLLM (Linux, CUDA GPU): pip install 'frida-decisions[vllm]'") from error
        from .vllm_backend.engine import VllmJudge
        return VllmJudge
    if name == "MlxJudge":
        try:
            import mlx.core  # noqa: F401
        except ImportError as error:
            raise ImportError("MlxJudge needs MLX on Apple Silicon: pip install 'frida-decisions[mlx]'") from error
        from .mlx_backend import MlxJudge
        return MlxJudge
    raise AttributeError(name)
