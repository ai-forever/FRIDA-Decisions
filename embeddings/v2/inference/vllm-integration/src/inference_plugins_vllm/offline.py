"""Joint offline embeddings using the public token embedding API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from inference_plugins.heads import unpack
from vllm.pooling_params import PoolingParams

from .checkpoint import load_checkpoint

if TYPE_CHECKING:
    from inference_plugins.types import JointEmbedding
    from vllm import LLM
    from vllm.inputs import PromptType


def joint_pooling_params(params: PoolingParams | None = None) -> PoolingParams:
    """Flag a native token request while preserving the caller's parameters."""
    params = params.clone() if params is not None else PoolingParams()
    if params.dimensions is not None or params.use_activation:
        raise ValueError("Embedding heads use their saved dimensions and normalization")
    params.task = "token_embed"
    params.extra_kwargs = {**(params.extra_kwargs or {}), "joint_embeddings": True}
    params.skip_reading_prefix_cache = True
    params.use_activation = False
    return params


def encode_joint(
    llm: LLM,
    prompts: PromptType | Sequence[PromptType],
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    **kwargs: Any,
) -> list[JointEmbedding]:
    """Return both named heads per prompt from an LLM configured for token_embed."""
    if llm.model_config.architecture != "Qwen35JointEmbedding":
        raise ValueError("encode_joint requires the Qwen35JointEmbedding architecture")
    contract = load_checkpoint(llm.llm_engine.vllm_config).contract
    outputs = llm.encode(
        prompts,
        pooling_task="token_embed",
        pooling_params=[joint_pooling_params(p) for p in pooling_params]
        if isinstance(pooling_params, Sequence)
        else joint_pooling_params(pooling_params),
        **kwargs,
    )
    return [unpack(item.outputs.data, contract) for item in outputs]
