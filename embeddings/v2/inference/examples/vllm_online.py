r"""Call /pooling using native single-head tasks or combined plugin output.

    python examples/vllm_online.py --url http://localhost:8000 --model joint-model \
        --task embed --texts 'Example query'

Use --task token_embed for multivectors or --task plugin for both heads.
The client sends each text as a user message with the generation prompt.
"""

from __future__ import annotations

import json

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    arguments,
    display,
    post_json,
)


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    args = arguments(__doc__, online=True, pooling_tasks=True)
    payload = {"model": args.model, "task": args.task, "encoding_format": args.encoding_format}
    if args.data_envelope and args.task != "plugin":
        raise ValueError("--data-envelope requires --task plugin")
    items = []
    for text in args.texts:
        chat = {
            "messages": [{"role": "user", "content": text}],
            "add_generation_prompt": True,
        }
        request = {**payload, "data": chat} if args.data_envelope else {**payload, **chat}
        items.extend(post_json(args.url.rstrip("/") + "/pooling", request)["data"])
    response = {"data": items}
    if args.encoding_format == "float":
        if args.task == "plugin":
            display(response["data"])
        else:
            field = "dense" if args.task == "embed" else "multivector"
            display([{field: item["data"]} for item in response["data"]])
    else:
        print(json.dumps(response, indent=2))  # noqa: T201 - Example CLI output.


if __name__ == "__main__":
    main()
