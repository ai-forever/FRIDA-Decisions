"""Build and save a materialized joint model with native Sentence Transformers modules."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from inference_plugins.checkpoint import JointCheckpoint
from inference_plugins.contract import Contract

MATRIX_NDIM = 2
DEFAULT_MAX_LENGTH = 4096


def convert(
    checkpoint: Path,
    output: Path,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    query_prefix: str = "",
    document_prefix: str = "",
    local_files_only: bool = False,
) -> Path:
    """Merge the backbone, construct native ST modules, and save the complete model."""
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules import Dense, Normalize, Transformer
    from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
    from sentence_transformers.sentence_transformer.modules import Pooling

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if checkpoint == output or checkpoint in output.parents:
        raise ValueError("Output must be outside the source checkpoint")
    settings = json.loads((checkpoint / "multivector_config.json").read_text())
    if not settings.get("use_projection"):
        raise ValueError("This joint export requires a trained projection")
    from safetensors.torch import load_file

    weights = load_file(str(checkpoint / "multivector_projection.safetensors"))
    if set(weights) != {"weight"} or weights["weight"].ndim != MATRIX_NDIM:
        raise ValueError("Expected a bias-free projection with exactly a matrix named weight")
    weight = weights["weight"]
    contract = Contract(
        dense_dim=weight.shape[1],
        multivector_dim=settings["embedding_size"],
        max_length=max_length,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        use_chat_template=True,
    )
    if weight.shape[0] != contract.multivector_dim:
        raise ValueError("Projection dimensions disagree with multivector_config.json")
    adapter_path = checkpoint / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text()) if adapter_path.exists() else None
    backbone = adapter["base_model_name_or_path"] if adapter else str(checkpoint)
    transformer = Transformer(
        backbone,
        max_seq_length=max_length,
        tokenizer_name_or_path=str(checkpoint),
        model_kwargs={"dtype": torch.bfloat16, "local_files_only": local_files_only},
        config_kwargs={
            "local_files_only": local_files_only,
            "revision": adapter.get("revision") if adapter else None,
        },
        processor_kwargs={
            "local_files_only": True,
            "max_length": max_length,
            "model_max_length": max_length,
        },
        processing_kwargs={
            "chat_template": {"add_generation_prompt": True, "restore_suffix": True}
        },
        modality_config={
            "text": {"method": "forward", "method_output_name": "last_hidden_state"},
            "image": {"method": "forward", "method_output_name": "last_hidden_state"},
            "message": {
                "method": "forward",
                "method_output_name": "last_hidden_state",
                "format": "structured",
            },
        },
        module_output_name="token_embeddings",
    )
    if adapter:
        transformer.model = PeftModel.from_pretrained(
            transformer.model, str(checkpoint), local_files_only=True
        ).merge_and_unload(safe_merge=True)
    if transformer.model.config.model_type != "qwen3_5":
        raise ValueError("Expected the full multimodal Qwen3.5 backbone")
    if transformer.get_embedding_dimension() != contract.dense_dim:
        raise ValueError("Projection input dimension disagrees with the backbone")
    model = SentenceTransformer(
        modules=[
            transformer,
            Pooling(contract.dense_dim, pooling_mode="lasttoken", include_prompt=True),
            Dense(
                contract.dense_dim,
                contract.multivector_dim,
                bias=False,
                activation_function=torch.nn.Identity(),
                init_weight=weight,
                module_input_name="token_embeddings",
                module_output_name="multivector_embeddings",
            ),
            Normalize(module_input_name="sentence_embedding"),
            Normalize(module_input_name="multivector_embeddings"),
            MultiVectorMask(skiplist_words=[]),
        ],
        device="cpu",
        prompts={"query": query_prefix, "document": document_prefix},
        similarity_fn_name="cosine",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".joint-export-", dir=output.parent) as temp:
        staging = Path(temp) / "model"
        model.save_pretrained(str(staging), safe_serialization=True)
        exported = JointCheckpoint.load(staging)
        torch.testing.assert_close(
            exported.load_projection(), weight.to(model.dtype), rtol=0, atol=0
        )
        if (staging / "adapter_config.json").exists():
            raise ValueError("Expected materialized backbone weights, not an adapter-only save")
        staging.rename(output)
    return output


def main() -> None:
    """Export one ST repository usable directly by ST and the engine plugins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(  # noqa: T201 - CLI output.
        convert(
            args.checkpoint,
            args.output,
            max_length=args.max_length,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
            local_files_only=args.local_files_only,
        )
    )


if __name__ == "__main__":
    main()
