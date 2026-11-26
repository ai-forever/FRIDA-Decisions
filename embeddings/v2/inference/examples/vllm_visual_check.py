r"""Check visual inference with generated images; no image files or downloads needed.

Offline, inside a vLLM environment::

    python examples/vllm_visual_check.py --model /path/to/merged-st-model

Against a running server (standard-library-only client)::

    python examples/vllm_visual_check.py --url http://localhost:8000 --model joint-model

The test encodes text, an image, image+text, and a changed image with identical
text. It checks finite embeddings, visual-token rows, and changes in both heads
when only pixels change. This checks execution, not semantic retrieval quality.
Use a complete multimodal export and a context length that fits image patches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from _common import engine_contract, post_json  # noqa: PLC2701 - Shared sibling example helper.
from _visual import (  # noqa: PLC2701 - Shared visual smoke checks.
    offline_visual_inputs,
    validate_visual_outputs,
    visual_inputs,
)

if TYPE_CHECKING:
    from inference_plugins.types import JointEmbedding


def main() -> None:
    """Run the same controlled visual test offline or over HTTP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local model path offline; served model name online"
    )
    parser.add_argument("--url", help="Server URL; omit to load the model locally")
    parser.add_argument("--result", type=Path, help="Optional file for complete embeddings")
    args = parser.parse_args()
    if args.url:
        items: list[JointEmbedding] = [
            post_json(
                args.url.rstrip("/") + "/pooling",
                {
                    "model": args.model,
                    "task": "plugin",
                    "messages": messages,
                    "add_generation_prompt": False,
                },
            )["data"][0]
            for messages in visual_inputs()
        ]
    else:
        from inference_plugins_vllm.offline import encode_joint
        from vllm import LLM
        from vllm.config import PoolerConfig

        root, contract, tokenizer = engine_contract(args.model)
        llm = LLM(
            model=str(root),
            runner="pooling",
            pooler_config=PoolerConfig(task="token_embed"),
            hf_overrides={"architectures": ["Qwen35JointEmbedding"]},
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_model_len=contract.max_length,
            enforce_eager=True,
        )
        items = encode_joint(llm, offline_visual_inputs(tokenizer))
    report = validate_visual_outputs(items)
    if args.result:
        args.result.write_text(json.dumps(items) + "\n")
    print(json.dumps(report, indent=2))  # noqa: T201 - Example CLI output.
    print("PASS: image-only, image+text, and changed pixels affect both embedding heads")  # noqa: T201 - Example CLI output.


if __name__ == "__main__":
    main()
