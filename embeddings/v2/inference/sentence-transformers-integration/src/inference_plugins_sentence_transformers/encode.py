"""Run both Sentence Transformers heads in one batched forward pass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from sentence_transformers.util import batch_to_device

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentence_transformers import SentenceTransformer


@torch.inference_mode()
def encode_joint(
    model: SentenceTransformer,
    inputs: Sequence[Any],
    *,
    role: Literal["query", "document"] = "document",
    batch_size: int = 32,
    processing_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Return dense and unpadded token vectors in input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prompt = model.prompts.get(role, model.prompts.get(model.default_prompt_name or ""))
    model.eval()
    results = []
    for start in range(0, len(inputs), batch_size):
        batch = list(inputs[start : start + batch_size])
        features = model.preprocess(
            batch, prompt=prompt, task=role, processing_kwargs=processing_kwargs or {}
        )
        output = model(batch_to_device(features, model.device), task=role)
        for dense, tokens, mask in zip(
            output["sentence_embedding"],
            output["multivector_embeddings"],
            output["attention_mask"],
            strict=True,
        ):
            results.append(
                {"dense": dense.detach().cpu(), "multivector": tokens[mask.bool()].detach().cpu()}
            )
    return results
