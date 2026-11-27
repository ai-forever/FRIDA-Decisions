r"""Call native SGLang /encode with pretokenized inputs and both heads.

Run from the repository root on a supported Linux/accelerator host.
Install the integration::

    UV_CACHE_DIR=.uv-cache uv sync --project sglang-integration

Start the server, setting external model discovery before importing SGLang::

    SGLANG_EXTERNAL_MODEL_PACKAGE=inference_plugins_sglang.models \\
    uv run --project sglang-integration python -m inference_plugins_sglang.serve \\
        --model-path /path/to/st-model \\
        --json-model-override-args '{"architectures":["Qwen35JointEmbedding"]}' \\
        --is-embedding \\
        --disable-radix-cache \\
        --chunked-prefill-size -1 \\
        --context-length 512 \\
        --port 30000

In another terminal, use the same checkpoint/tokenizer locally::

    uv run --project sglang-integration python examples/sglang_online.py \\
        --url http://localhost:30000 --model /path/to/st-model \\
        --role document --texts "Semantic search uses meaning."

Use the complete Sentence Transformers folder created by joint-st-convert
directly. The architecture override selects our plugin; ST modules and the
multimodal processor are preserved. No engine export is needed. The launcher
converts raw tensors to lists at the HTTP boundary.
GPU serving still needs verification in the target environment. The client normalizes raw pooled states into
the dense output. See examples/README.md for details.
"""

from __future__ import annotations

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    arguments,
    display,
    engine_contract,
    post_json,
    sglang_outputs,
)


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    args = arguments(__doc__, online=True)
    # Use the server checkpoint to apply role prefixes and preserve the chat suffix.
    _, contract, tokenizer = engine_contract(args.model)
    response = post_json(
        args.url.rstrip("/") + "/encode",
        {
            "input_ids": contract.tokenize(tokenizer, args.texts, args.role),
            "return_pooled_hidden_states": True,
        },
    )
    display(sglang_outputs(response, contract))


if __name__ == "__main__":
    main()
