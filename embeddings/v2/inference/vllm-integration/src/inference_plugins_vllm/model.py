"""Targets the vLLM pooler API in the handoff's local checkout."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import torch
from inference_plugins.heads import pack
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise.heads import TokenEmbeddingPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler as NativeTokenPooler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import StageMissingLayer, no_init_weights
from vllm.multimodal import MULTIMODAL_REGISTRY

from .checkpoint import load_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Set as AbstractSet

    from inference_plugins.contract import Contract
    from vllm.config import VllmConfig
    from vllm.pooling_params import PoolingParams
    from vllm.tasks import PoolingTask
    from vllm.v1.pool.metadata import PoolingMetadata


def _head_params(params: PoolingParams, normalize: bool) -> PoolingParams:
    if params.dimensions is not None:
        raise ValueError("Joint model does not support dimension truncation")
    if getattr(params, "late_chunking_params", None) is not None:
        raise ValueError("Joint model does not support late chunking")
    result = copy(params)
    result.use_activation = normalize
    return result


class DensePooler(Pooler):
    """Native last-token pooling and embedding head with contract defaults."""

    def __init__(self, contract: Contract) -> None:
        super().__init__()
        self.contract = contract
        self.pooling = LastPool()
        self.head = EmbeddingPoolerHead(
            activation=PoolerNormalize() if contract.normalize_dense else None
        )

    def get_supported_tasks(self) -> AbstractSet[PoolingTask]:  # noqa: PLR6301 - Engine interface.
        """Expose dense embedding execution."""
        return {"embed"}

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[torch.Tensor | None]:
        """Compute dense heads without accumulating token states."""
        metadata = copy(pooling_metadata)
        metadata.pooling_params = []
        for params in pooling_metadata.pooling_params:
            if (params.extra_kwargs or {}).get("joint_embeddings"):
                raise ValueError("Joint output requires the token_embed execution task")
            activation = (
                self.contract.normalize_dense
                if params.use_activation is None
                else params.use_activation
            )
            metadata.pooling_params.append(_head_params(params, activation))
        dense = self.head(self.pooling(hidden_states, pooling_metadata), metadata)
        finished = pooling_metadata.get_pooling_cursor().get_finished_mask()
        return [row.clone() if done else None for row, done in zip(dense, finished, strict=True)]


class TokenPooler(NativeTokenPooler):
    """Use native batched token projection, then apply the saved token filter."""

    def __init__(self, contract: Contract, projection: nn.Linear) -> None:
        super().__init__(
            pooling=AllPool(),
            head=TokenEmbeddingPoolerHead(
                head_dtype=projection.weight.dtype,
                projector=projection,
                activation=PoolerNormalize() if contract.normalize_multivector else None,
            ),
        )
        self.contract = contract

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        """Require original IDs only when token filtering is configured."""
        return super().get_pooling_updates(task) | PoolingParamsUpdate(
            requires_token_ids=bool(self.contract.skip_token_ids)
        )

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[torch.Tensor | None]:
        """Keep native batching/chunk handling and normalize with saved defaults."""
        metadata = copy(pooling_metadata)
        metadata.pooling_params = [
            _head_params(
                p,
                self.contract.normalize_multivector
                if p.use_activation is None
                else p.use_activation,
            )
            for p in pooling_metadata.pooling_params
        ]
        # AllPool can split projected rows directly for single-step prompts. A
        # request that spans steps must keep the native hidden-state cache layout.
        outputs = None
        if (
            self.pooling.enable_chunked_prefill
            and metadata.pooling_params
            and all(metadata.get_pooling_cursor().get_finished_mask())
            and not any(state.hidden_states_cache for state in metadata.pooling_states)
        ):
            outputs = self.head.forward_batched(
                hidden_states, metadata.pooling_params, lambda rows: self.pooling(rows, metadata)
            )
        if outputs is None:
            outputs = super().forward(hidden_states, metadata)
        if not self.contract.skip_token_ids:
            return outputs
        ids = pooling_metadata.get_prompt_token_ids_cpu()
        filtered: list[torch.Tensor | None] = []
        for tokens, token_ids in zip(outputs, ids, strict=True):
            if tokens is None:
                filtered.append(None)
                continue
            mask = torch.ones(len(token_ids), device=tokens.device, dtype=torch.bool)
            ids_tensor = torch.as_tensor(token_ids, device=tokens.device)
            for skip in self.contract.skip_token_ids:
                mask &= ids_tensor != skip
            filtered.append(tokens[mask])
        return filtered


class JointPooler(Pooler):
    """Compose registered dense and token poolers, following BGE-M3's design."""

    def __init__(self, token_pooler: TokenPooler, embed_pooler: DensePooler) -> None:
        super().__init__()
        self.token_pooler = token_pooler
        self.embed_pooler = embed_pooler
        self.contract = token_pooler.contract

    def get_supported_tasks(self) -> AbstractSet[PoolingTask]:  # noqa: PLR6301 - Engine interface.
        """Use the native V2 token task for both token-only and flagged joint output."""
        return {"token_embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        """Combine the requirements of both registered poolers."""
        return self.embed_pooler.get_pooling_updates(
            "embed"
        ) | self.token_pooler.get_pooling_updates("token_embed")

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[torch.Tensor | None]:
        """Reuse one backbone pass and pack both heads only for joint requests."""
        flags = [
            bool((p.extra_kwargs or {}).get("joint_embeddings"))
            for p in pooling_metadata.pooling_params
        ]
        token_metadata = copy(pooling_metadata)
        token_metadata.pooling_params = [
            _head_params(p, self.contract.normalize_multivector) if joint else p
            for p, joint in zip(pooling_metadata.pooling_params, flags, strict=True)
        ]
        tokens = self.token_pooler(hidden_states, token_metadata)
        if not any(joint and row is not None for joint, row in zip(flags, tokens, strict=True)):
            return tokens
        dense_metadata = copy(pooling_metadata)
        dense_metadata.pooling_params = []
        for params in pooling_metadata.pooling_params:
            dense_params = _head_params(params, self.contract.normalize_dense)
            dense_params.extra_kwargs = {
                key: value
                for key, value in (params.extra_kwargs or {}).items()
                if key != "joint_embeddings"
            }
            dense_metadata.pooling_params.append(dense_params)
        dense = self.embed_pooler(hidden_states, dense_metadata)
        return [
            pack(vector, row, self.contract) if joint and row is not None else row
            for vector, row, joint in zip(dense, tokens, flags, strict=True)
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen35JointEmbedding(Qwen3_5ForConditionalGeneration):
    """Run the Qwen3.5 backbone with the trained joint embedding heads."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Disable prefix caching for joint embeddings")
        # Use the same placeholders as vLLM's pooling adapter, avoiding retained
        # generation-only parameters and handling tied embeddings natively.
        with no_init_weights(
            self,
            lambda layer: StageMissingLayer("output", layer),
            targets=(LogitsProcessor, ParallelLMHead),
        ):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        checkpoint = load_checkpoint(vllm_config, with_projection=True)
        c = checkpoint.contract
        weight = checkpoint.load_projection().to(device=next(self.parameters()).device)
        projection = nn.Linear(c.dense_dim, c.multivector_dim, bias=False, device="meta")
        projection.load_state_dict({"weight": weight}, assign=True)
        self.joint_pooler = JointPooler(TokenPooler(c, projection), DensePooler(c))
        self.pooler = DispatchPooler(
            {
                "embed": self.joint_pooler.embed_pooler,
                "token_embed": self.joint_pooler,
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone weights while preserving the ST projection."""
        # ST's AutoModel save uses bare language_model.* backbone names.
        weights = (
            ("model." + name if name.startswith("language_model.") else name, tensor)
            for name, tensor in weights
        )
        loaded = super().load_weights(weights)
        # The ST projection is loaded explicitly during construction.
        loaded.add("joint_pooler.token_pooler.head.projector.weight")
        return loaded
