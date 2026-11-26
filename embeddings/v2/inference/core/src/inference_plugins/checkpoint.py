"""Read our joint Sentence Transformers pipeline without importing ST."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from safetensors.torch import load_file

from inference_plugins.contract import Contract

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def _config(root: Path, path: str) -> dict[str, Any]:
    return json.loads((root / path).read_text())


def _modules(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "modules.json").read_text())


def _module_path(root: Path, module: dict[str, Any]) -> Path:
    path = (root / module["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("ST module path must be inside the checkpoint")
    return path


EXTRA_SETTINGS = ("version", "append_eos", "add_special_tokens", "skip_token_ids")


@dataclass(frozen=True)
class JointCheckpoint:
    """One validated ST pipeline and its trained token-projection location."""

    contract: Contract
    projection_path: Path

    @classmethod
    def load(cls, root: Path) -> JointCheckpoint:
        """Parse the ST pipeline once, requiring modules.json."""
        return _read_pipeline(root)

    def load_projection(self) -> torch.Tensor:
        """Load the sole bias-free ST token projection."""
        weights = load_file(str(self.projection_path))
        if set(weights) != {"linear.weight"}:
            raise ValueError("ST token projection must contain exactly linear.weight")
        weight = weights["linear.weight"]
        if tuple(weight.shape) != (self.contract.multivector_dim, self.contract.dense_dim):
            raise ValueError("ST token projection shape disagrees with its configuration")
        return weight


def _read_pipeline(root: Path) -> JointCheckpoint:
    modules = _modules(root)
    if modules and modules[-1]["type"].endswith(".MultiVectorMask"):
        mask = _config(_module_path(root, modules[-1]), "config.json")
        if set(mask) - {"skiplist_words", "skiplist_tasks", "keep_only_token_ids"} or (
            mask.get("skiplist_words") or mask.get("keep_only_token_ids")
        ):
            raise ValueError("Only an unfiltered MultiVectorMask is supported by the engines")
        modules = modules[:-1]
    kinds = [module["type"].rsplit(".", 1)[-1] for module in modules]
    if kinds[:3] != ["Transformer", "Pooling", "Dense"] or any(
        kind != "Normalize" for kind in kinds[3:]
    ):
        raise ValueError(
            "Unsupported ST module pipeline; expected Transformer, Pooling, Dense, then Normalize modules"
        )
    dense = [m for m in modules if m["type"].endswith(".Dense")]
    pooling = [m for m in modules if m["type"].endswith(".Pooling")]
    if len(dense) != 1 or len(pooling) != 1:
        raise ValueError("Expected one ST token Dense and one last-token Pooling module")
    head = _config(_module_path(root, dense[0]), "config.json")
    pool = _config(_module_path(root, pooling[0]), "config.json")
    if (
        head.get("bias", True)
        or not head.get("activation_function", "").endswith(".Identity")
        or head.get("module_input_name") != "token_embeddings"
        or head.get("module_output_name") != "multivector_embeddings"
    ):
        raise ValueError("Unsupported ST joint pooling/projection configuration")
    if (
        pool.get("pooling_mode") != "lasttoken"
        or pool.get("embedding_dimension") != head["in_features"]
        or pool.get("include_prompt", True) is not True
    ):
        raise ValueError("Unsupported ST dense pooling configuration")
    normalized = set()
    for module in modules:
        if module["type"].endswith(".Normalize"):
            config = _config(_module_path(root, module), "config.json")
            source = config.get("module_input_name", "sentence_embedding")
            if source != config.get("module_output_name", source):
                raise ValueError("Normalize must preserve its embedding field")
            normalized.add(source)
    transformer = _config(root, "sentence_bert_config.json")
    settings = _config(root, "config_sentence_transformers.json")
    prompts = settings.get("prompts", {})
    values = (
        _config(root, "config.json").get("joint_embeddings", {})
        if (root / "config.json").exists()
        else {}
    )
    if set(values) - set(EXTRA_SETTINGS):
        raise ValueError(
            "joint_embeddings accepts only version, append_eos, add_special_tokens and skip_token_ids; other settings belong in ST modules"
        )
    inferred = dict(
        dense_dim=head["in_features"],
        multivector_dim=head["out_features"],
        normalize_dense="sentence_embedding" in normalized,
        normalize_multivector="multivector_embeddings" in normalized,
        max_length=_config(root, "tokenizer_config.json")["model_max_length"],
        use_chat_template="message" in transformer.get("modality_config", {}),
        query_prefix=prompts.get("query", ""),
        document_prefix=prompts.get("document", ""),
    )
    return JointCheckpoint(
        Contract.from_dict({**values, **inferred}),
        _module_path(root, dense[0]) / "model.safetensors",
    )
