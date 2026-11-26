"""Load the joint ST model and return dense plus unpadded multivector embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    arguments,
    display,
)

if TYPE_CHECKING:
    from torch import Tensor


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    args = arguments(__doc__, default_model="ai-forever/FRIDA-Embeddings-V2-2B")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    encode = model.encode_query if args.role == "query" else model.encode_document
    features = cast("list[dict[str, Tensor]]", encode(args.texts, output_value=None))
    display(
        [
            {
                "dense": item["sentence_embedding"].tolist(),
                "multivector": item["multivector_embeddings"][
                    item["attention_mask"].bool()
                ].tolist(),
            }
            for item in features
        ]
    )


if __name__ == "__main__":
    main()
