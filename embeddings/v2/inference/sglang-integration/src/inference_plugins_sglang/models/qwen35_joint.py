"""Full Qwen3.5 vision/language backbone with joint retrieval pooling."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from inference_plugins.checkpoint import JointCheckpoint
from inference_plugins.heads import JointHeads
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.runtime_context import get_model
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from transformers import PretrainedConfig


class JointPooler(nn.Module):
    """Pool complete multimodal prompt states without changing native vision handling."""

    def __init__(self, heads: JointHeads) -> None:
        super().__init__()
        self.heads = heads

    def forward(self, hidden: torch.Tensor, forward_batch: ForwardBatch) -> EmbeddingPoolerOutput:
        """Return token vectors and raw last-token states for the shared client decoder."""
        lengths = forward_batch.extend_seq_lens_cpu
        if any(forward_batch.extend_prefix_lens_cpu):
            raise ValueError("Joint embeddings require full prompts; disable radix cache")
        ids = forward_batch.input_ids
        if sum(lengths) != ids.numel() or list(lengths) != forward_batch.seq_lens.tolist():
            raise ValueError("Joint embeddings require full prompts; disable chunked prefill")
        parts = hidden.split(lengths)
        pairs = [self.heads(h, tokens) for h, tokens in zip(parts, ids.split(lengths), strict=True)]
        return EmbeddingPoolerOutput(
            embeddings=[tokens for _, tokens in pairs],
            pooled_hidden_states=torch.stack([h[-1] for h in parts])
            if forward_batch.return_pooled_hidden_states
            else None,
        )


class Qwen35JointEmbedding(Qwen3_5ForConditionalGeneration):
    """Preserve native Qwen3.5 image processing and replace only embedding pooling."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        checkpoint = JointCheckpoint.load(Path(get_model().model_path))
        self.pooler = JointPooler(
            JointHeads(
                checkpoint.contract,
                checkpoint.load_projection(),
                device=next(self.parameters()).device,
            )
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = True,
        **kwargs: Any,
    ) -> EmbeddingPoolerOutput:
        """Let SGLang embed image patches and run the language model before our pooler."""
        if not get_embedding:
            raise ValueError("Joint model only supports embeddings")
        return super().forward(input_ids, positions, forward_batch, get_embedding=True, **kwargs)


EntryClass = Qwen35JointEmbedding
