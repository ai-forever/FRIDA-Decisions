r"""Encode an image plus text through our native Qwen3.5 multimodal backbone.

    python examples/vllm_multimodal_online.py --url http://localhost:8000 \\
        --model joint-model --image page.png --text 'Describe this page' --task plugin

task selects embed, token_embed, or plugin (both heads).
Serve with VLLM_PLUGINS=joint_embeddings to enable the pooling endpoint. Choose a max-model-len large enough for the image patches.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

from _common import display, post_json  # noqa: PLC2701 - Shared sibling example helper.


def main() -> None:
    """Send a native image conversation and display the selected embedding heads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="joint-model")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", default="")
    parser.add_argument(
        "--task",
        choices=["embed", "token_embed", "plugin"],
        default="plugin",
    )
    parser.add_argument("--encoding-format", choices=["float", "base64"], default="float")
    parser.add_argument(
        "--data-envelope", action="store_true", help="Use the official plugin data envelope"
    )
    args = parser.parse_args()
    if args.data_envelope and args.task != "plugin":
        parser.error("--data-envelope requires --task plugin")
    mime = mimetypes.guess_type(args.image.name)[0]
    if mime is None or not mime.startswith("image/"):
        parser.error("--image must have a recognized image extension")
    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": args.text},
            ],
        }
    ]
    payload = {"model": args.model, "task": args.task, "encoding_format": args.encoding_format}
    if args.data_envelope:
        payload["data"] = {"messages": messages, "add_generation_prompt": False}
    else:
        payload["messages"] = messages
        payload["add_generation_prompt"] = False
    response = post_json(args.url.rstrip("/") + "/pooling", payload)
    if args.encoding_format == "float":
        display(response["data"])
    else:
        print(json.dumps(response, indent=2))  # noqa: T201 - Example CLI output.


if __name__ == "__main__":
    main()
