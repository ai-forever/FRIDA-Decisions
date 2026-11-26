"""Explicit preprocessing and head contract, stored alongside exported weights."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .types import ChatTokenizer, PathLike, Role, Tokenizer


@dataclass(frozen=True)
class Contract:
    """Validated preprocessing settings and dimensions for the joint embedding heads."""

    version: int = 1
    dense_dim: int = 2048
    multivector_dim: int = 128
    normalize_dense: bool = True
    normalize_multivector: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    append_eos: bool = False
    add_special_tokens: bool = False
    use_chat_template: bool = False
    max_length: int = 512
    skip_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("Unsupported joint layout version")
        if (
            self.dense_dim <= 0
            or self.multivector_dim <= 0
            or self.dense_dim % self.multivector_dim
        ):
            raise ValueError("Dense dimension must be a positive multiple of multivector dimension")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if not all(isinstance(i, int) and i >= 0 for i in self.skip_token_ids):
            raise ValueError("skip_token_ids must contain nonnegative integers")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a validated contract from serialized settings."""
        data = dict(data)
        data["skip_token_ids"] = tuple(data.get("skip_token_ids", ()))
        return cls(**data)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        """Load the embedding contract from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: PathLike) -> None:
        """Write the embedding contract as UTF-8 JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def tokenize(
        self, tokenizer: Tokenizer, texts: list[str], role: Role = "document"
    ) -> list[list[int]]:
        """Apply role prefixes, truncation, and optional EOS to a batch of texts."""
        if role not in {"query", "document"}:
            raise ValueError("role must be query or document")
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be a nonempty list of strings")
        prefix = self.query_prefix if role == "query" else self.document_prefix
        budget = self.max_length - int(self.append_eos)
        if budget < 1:
            raise ValueError("max_length must leave room for content and EOS")
        prepared = [prefix + text for text in texts]
        if self.use_chat_template:
            prepared = [
                cast("ChatTokenizer", tokenizer).apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for text in prepared
            ]
        ids = tokenizer(
            prepared,
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            max_length=budget,
        )["input_ids"]
        if self.append_eos:
            if tokenizer.eos_token_id is None:
                raise ValueError("EOS requested but tokenizer has no EOS")
            ids = [row + [tokenizer.eos_token_id] for row in ids]
        if any(not row for row in ids):
            raise ValueError("Empty token sequences are unsupported")
        return ids
