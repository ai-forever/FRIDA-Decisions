"""Run the external SGLang joint model without an HTTP server."""

from __future__ import annotations

import os

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    arguments,
    display,
    engine_contract,
    sglang_outputs,
)


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    args = arguments(__doc__)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "inference_plugins_sglang.models"
    os.environ["SGLANG_EXTERNAL_MM_MODEL_ARCH"] = "Qwen35JointEmbedding"
    os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = "inference_plugins_sglang.processors"
    root, contract, tokenizer = engine_contract(args.model)
    from sglang import Engine
    from sglang.srt.managers.io_struct import EmbeddingReqInput

    engine = Engine(
        model_path=str(root),
        json_model_override_args='{"architectures":["Qwen35JointEmbedding"]}',
        is_embedding=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        context_length=contract.max_length,
    )
    try:
        # This checkout's public Engine.encode() does not forward
        # return_pooled_hidden_states. Use its native request path explicitly.
        # This adapter depends on SGLang internals; see examples/README.md.
        request = EmbeddingReqInput(
            input_ids=contract.tokenize(tokenizer, args.texts, args.role),
            return_pooled_hidden_states=True,
        )
        generator = engine.tokenizer_manager.generate_request(request, None)
        try:
            response = engine.loop.run_until_complete(anext(generator))
        finally:
            engine.loop.run_until_complete(generator.aclose())
        display(sglang_outputs(response, contract))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
