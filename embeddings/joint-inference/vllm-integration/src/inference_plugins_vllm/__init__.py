"""Engine imports are lazy so installing the package does not require CUDA."""

from __future__ import annotations


def register_vllm() -> None:
    """Register the joint embedding architecture with vLLM."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )

    MODELS_CONFIG_MAP["Qwen35JointEmbedding"] = Qwen3_5ForConditionalGenerationConfig

    ModelRegistry.register_model(
        "Qwen35JointEmbedding", "inference_plugins_vllm.model:Qwen35JointEmbedding"
    )


def register_io_processor() -> str:
    # IOProcessor resolution uses a dotted qualified name; ModelRegistry
    # above uses its separate module:class convention.
    """Return the import path for the joint embedding IO processor."""
    return "inference_plugins_vllm.io.JointIOProcessor"
