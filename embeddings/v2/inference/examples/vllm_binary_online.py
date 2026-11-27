"""Request combined heads using vLLM's standard output formats.

    python examples/vllm_binary_online.py --url http://localhost:8000 --encoding-format bytes

Use float, base64, bytes, or bytes_only with any supported launcher.
This client needs only the Python standard library; bytes metadata describes
how to reconstruct the tensors. bytes_only deliberately omits that metadata.
"""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main() -> None:
    """Send a joint request and print the named output shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="joint-model")
    parser.add_argument("--texts", nargs="+", default=["Example document", "Another document"])
    parser.add_argument(
        "--encoding-format", choices=["float", "base64", "bytes", "bytes_only"], default="float"
    )
    args = parser.parse_args()
    if urlsplit(args.url).scheme not in {"http", "https"}:
        parser.error("--url must use http or https")
    media_type = (
        "application/octet-stream"
        if args.encoding_format in {"bytes", "bytes_only"}
        else "application/json"
    )
    for text in args.texts:
        request_embedding(args, text, media_type)


def request_embedding(args: argparse.Namespace, text: str, media_type: str) -> None:
    """Send one chat-formatted document and print the returned shapes."""
    request = Request(  # noqa: S310 - Validated user-selected HTTP(S) inference server.
        args.url.rstrip("/") + "/pooling",
        data=json.dumps(
            {
                "model": args.model,
                "task": "plugin",
                "encoding_format": args.encoding_format,
                "embed_dtype": "float32",
                "endianness": "little",
                "messages": [{"role": "user", "content": text}],
                "add_generation_prompt": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json", "Accept": media_type},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:  # noqa: S310 - Validated user-selected HTTP(S) inference server.
        body = response.read()
        actual_type = response.headers.get_content_type()
        metadata = response.headers.get("metadata")
    if actual_type != media_type:
        raise RuntimeError(f"Requested {media_type}, got {actual_type}")
    if args.encoding_format in {"bytes", "bytes_only"}:
        print(f"Transport: {actual_type}; response bytes: {len(body)}")  # noqa: T201 - CLI output.
        if metadata is not None:
            print(json.dumps(json.loads(metadata), indent=2))  # noqa: T201 - Tensor layout for decoding.
        return
    result = json.loads(body)
    if len(result["data"]) != 1:
        raise ValueError("Expected one joint embedding per input")
    for index, item in enumerate(result["data"]):
        shapes = {}
        for name, values in item.items():
            if args.encoding_format == "base64":
                shapes[name] = tuple(values["shape"])
            elif name == "dense":
                shapes[name] = (len(values),)
            else:
                shapes[name] = (len(values), len(values[0]) if values else 0)
        print(f"Input {index}: " + ", ".join(f"{name}={shape}" for name, shape in shapes.items()))  # noqa: T201 - CLI output.
    print(f"Transport: {actual_type}; response bytes: {len(body)}")  # noqa: T201 - CLI output.


if __name__ == "__main__":
    main()
