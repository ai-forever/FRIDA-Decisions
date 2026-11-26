"""Small shared helpers for the runnable examples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inference_plugins.contract import Contract
    from inference_plugins.types import DisplayEmbedding, JointEmbedding, PathLike
    from transformers import PreTrainedTokenizerBase


def arguments(
    description: str | None,
    *,
    online: bool = False,
    default_model: str | None = None,
    pooling_tasks: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        required=default_model is None,
        default=default_model,
        help="Local model directory (see examples/README.md)",
    )
    parser.add_argument(
        "--texts", nargs="+", default=["What is semantic search?", "Search using meaning."]
    )
    if not pooling_tasks:
        parser.add_argument("--role", choices=["query", "document"], default="document")
    if pooling_tasks:
        parser.add_argument(
            "--task",
            choices=["embed", "token_embed", "plugin"],
            default="plugin",
        )
    if pooling_tasks:
        parser.add_argument(
            "--data-envelope",
            action="store_true",
            help="Use the official data envelope (task=plugin only)",
        )
    if online and pooling_tasks:
        parser.add_argument("--encoding-format", choices=["float", "base64"], default="float")
    if online:
        parser.add_argument(
            "--url", required=True, help="Server base URL, e.g. http://localhost:8000"
        )
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> Any:  # noqa: ANN401 - dynamic external format
    request = Request(  # noqa: S310 - Explicit user-supplied inference server URL.
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:  # noqa: S310 - Explicit user-supplied inference server URL.
        return json.load(response)


def engine_contract(model: PathLike) -> tuple[Path, Contract, PreTrainedTokenizerBase]:
    from inference_plugins.checkpoint import JointCheckpoint
    from transformers import AutoTokenizer

    root = Path(model).resolve()
    contract = JointCheckpoint.load(root).contract
    if (root / "adapter_config.json").exists() or not (root / "config.json").exists():
        raise ValueError(
            "Expected a materialized Sentence Transformers model; export with joint-st-convert"
        )
    return root, contract, AutoTokenizer.from_pretrained(root)


def response_shape(value: object) -> str:
    """Summarize response structure without dumping embedding values."""
    if hasattr(value, "shape"):
        return f"{type(value).__name__}{list(value.shape)}"
    if isinstance(value, dict):
        return f"object(keys={list(value)[:8]})"
    if isinstance(value, list):
        if not value:
            return "list[0]"
        return f"list[{len(value)}] of {response_shape(value[0])}"
    return type(value).__name__


def sglang_outputs(
    response: dict[str, Any] | list[dict[str, Any]], contract: Contract
) -> list[JointEmbedding]:
    """Expose named heads while preserving native pooled_hidden_state semantics."""
    items = response if isinstance(response, list) else [response]
    result: list[JointEmbedding] = []
    for item in items:
        if "pooled_hidden_state" not in item:
            raise ValueError(
                "SGLang response lacks pooled_hidden_state; request return_pooled_hidden_states=True"
            )
        dense, tokens = item["pooled_hidden_state"], item["embedding"]
        dense = dense.tolist() if hasattr(dense, "tolist") else dense
        tokens = tokens.tolist() if hasattr(tokens, "tolist") else tokens
        valid_dense = (
            isinstance(dense, list)
            and len(dense) == contract.dense_dim
            and all(isinstance(value, (int, float)) for value in dense)
        )
        valid_tokens = isinstance(tokens, list) and all(
            isinstance(row, list)
            and len(row) == contract.multivector_dim
            and all(isinstance(value, (int, float)) for value in row)
            for row in tokens
        )
        if not valid_dense or not valid_tokens:
            raise ValueError(
                f"Unexpected SGLang output: pooled_hidden_state={response_shape(dense)}, "
                f"embedding={response_shape(tokens)}; expected [{contract.dense_dim}] "
                f"and [tokens, {contract.multivector_dim}]. "
                "If pooled_hidden_state is an object rather than a numeric list, "
                "the server must convert the raw tensor with .tolist() before HTTP serialization."
            )
        if contract.normalize_dense:
            norm = max(math.sqrt(sum(value * value for value in dense)), 1e-12)
            dense = [value / norm for value in dense]
        result.append({"dense": dense, "multivector": tokens})
    return result


def display(items: Sequence[DisplayEmbedding]) -> None:
    for index, item in enumerate(items):
        shapes = []
        if "dense" in item:
            shapes.append(f"dense=[{len(item['dense'])}]")
        if "multivector" in item:
            tokens = item["multivector"]
            shapes.append(f"multivector=[{len(tokens)}, {len(tokens[0]) if tokens else 0}]")
        print(f"Input {index}: " + ", ".join(shapes))  # noqa: T201 - CLI output is intended for the user.
    print(json.dumps(items, indent=2))  # noqa: T201 - CLI output is intended for the user.
