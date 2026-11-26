"""Shared request, response, and tokenizer types for embedding integrations."""

from __future__ import annotations

from os import PathLike as OSPathLike
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

PathLike: TypeAlias = str | OSPathLike[str]
Role: TypeAlias = Literal["query", "document"]
JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


EmbedTask: TypeAlias = Literal["dense", "multivector", "dense&multivector"]


class TensorOutput(TypedDict, total=False):
    """Selected embedding heads kept on their original tensor device."""

    dense: torch.Tensor
    multivector: torch.Tensor


class JointEmbedding(TypedDict):
    """JSON-compatible dense vector and variable-length multivector matrix."""

    dense: list[float]
    multivector: list[list[float]]


class TensorEmbedding(TypedDict):
    """Dense and token heads retained as tensors for non-JSON transports."""

    dense: torch.Tensor
    multivector: torch.Tensor


class DisplayEmbedding(TypedDict, total=False):
    """One or both named embedding heads for example-client output."""

    dense: list[float]
    multivector: list[list[float]]


class Tokenizer(Protocol):
    """The text-tokenization interface shared by HF and local test tokenizers."""

    eos_token_id: int | None

    def __call__(self, texts: list[str], **kwargs: Any) -> Mapping[str, Any]:
        """Tokenize texts using the requested preprocessing options."""
        ...


class ChatTokenizer(Tokenizer, Protocol):
    """A tokenizer that can render the multimodal checkpoint's saved template."""

    eos_token: str | None

    def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> str:
        """Render structured text messages using the saved template."""
        ...
