"""Load the joint ST model and return dense plus unpadded multivector embeddings."""

from __future__ import annotations

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    arguments,
    display,
)
from inference_plugins_sentence_transformers.encode import encode_joint


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    args = arguments(__doc__, default_model="ai-forever/FRIDA-Embeddings-V2-2B")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    items = encode_joint(model, args.texts, role=args.role)
    display(
        [
            {"dense": item["dense"].tolist(), "multivector": item["multivector"].tolist()}
            for item in items
        ]
    )


if __name__ == "__main__":
    main()
