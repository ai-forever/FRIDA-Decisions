r"""Call /pooling using native single-head tasks or combined plugin output.

    python examples/vllm_online.py --url http://localhost:8000 --model joint-model \
        --task embed --texts 'Example query'

Use --task token_embed for multivectors or --task plugin for both heads.
All tasks use native input/messages preprocessing. Supply training prefixes or
chat formatting explicitly when required.
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
    if args.data_envelope:
        if args.task != "plugin":
            raise ValueError("--data-envelope requires --task plugin")
        payload["data"] = {"input": args.texts}
    else:
        payload["input"] = args.texts
    response = post_json(args.url.rstrip("/") + "/pooling", payload)
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
