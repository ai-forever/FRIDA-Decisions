"""Resolve Sentence Transformers artifacts alongside vLLM's backbone weights."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download, snapshot_download
from inference_plugins.checkpoint import JointCheckpoint

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def load_checkpoint(config: VllmConfig, *, with_projection: bool = False) -> JointCheckpoint:
    """Load local files or resolve the ST pipeline at the backbone's Hub revision."""
    model = config.model_config
    root = Path(model.model)
    if root.is_dir() or root.is_absolute() or model.model.startswith(("./", "../")):
        return JointCheckpoint.load(root)

    revision = getattr(model.hf_config, "_commit_hash", None) or model.revision
    cache_dir = config.load_config.download_dir
    root = Path(
        hf_hub_download(
            model.model, "config.json", revision=revision, cache_dir=cache_dir, token=model.hf_token
        )
    ).parent
    # Hub cache snapshots are named by commit, even when requested by branch or tag.
    try:
        checkpoint = JointCheckpoint.load(root)
    except FileNotFoundError:
        snapshot_download(
            model.model,
            revision=root.name,
            cache_dir=cache_dir,
            token=model.hf_token,
            allow_patterns=[
                "modules.json",
                "sentence_bert_config.json",
                "config_sentence_transformers.json",
                "tokenizer_config.json",
                "*/config.json",
            ],
        )
        checkpoint = JointCheckpoint.load(root)
    if with_projection:
        hf_hub_download(
            model.model,
            filename=checkpoint.projection_path.relative_to(root.resolve()).as_posix(),
            revision=root.name,
            cache_dir=cache_dir,
            token=model.hf_token,
        )
    return checkpoint
