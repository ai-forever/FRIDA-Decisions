"""Deterministic image fixtures and checks for the visual inference examples."""

from __future__ import annotations

import base64
import io
import math
import struct
import zlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inference_plugins.types import ChatTokenizer, JointEmbedding

LABELS = ("text-only", "image-only", "image+text", "changed-image+text")
IMAGE_SIZE = 256
MIN_IMAGE_DELTA = 1e-6


def image_data_url(*, changed: bool = False) -> str:
    """Create a small RGB PNG with identical geometry but different colors."""
    foreground = (20, 70, 230) if changed else (230, 40, 20)
    background = (240, 240, 240)
    rows = b"".join(
        b"\x00"
        + bytes(
            component
            for x in range(IMAGE_SIZE)
            for component in (foreground if (x // 32 + y // 32) % 2 else background)
        )
        for y in range(IMAGE_SIZE)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", IMAGE_SIZE, IMAGE_SIZE, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def visual_inputs(text: str = "Describe the image.") -> list[list[dict[str, Any]]]:
    """Build a mixed batch with a controlled change to image pixels only."""

    def conversation(*, caption: bool, changed: bool = False) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_data_url(changed=changed)}}
        ]
        if caption:
            content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    return [
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        conversation(caption=False),
        conversation(caption=True),
        conversation(caption=True, changed=True),
    ]


def offline_visual_inputs(tokenizer: ChatTokenizer) -> list[dict[str, Any]]:
    """Format native prompts explicitly and attach PIL image payloads offline."""
    from PIL import Image

    prompts = []
    for messages in visual_inputs():
        content = []
        images = []
        for part in messages[0]["content"]:
            if part["type"] == "image_url":
                png = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                with Image.open(io.BytesIO(png)) as source:
                    images.append(source.convert("RGB"))
                content.append({"type": "image"})
            else:
                content.append(part)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False
        )
        prompt: dict[str, Any] = {"prompt": formatted}
        if images:
            prompt["multi_modal_data"] = {"image": images}
        prompts.append(prompt)
    return prompts


def validate_visual_outputs(items: Sequence[JointEmbedding]) -> dict[str, object]:
    """Check shape/finite values and an observable response to changed pixels."""
    if len(items) != len(LABELS):
        raise ValueError("Expected one output per visual test input")
    dense_dim = len(items[0]["dense"])
    width = len(items[0]["multivector"][0]) if items[0]["multivector"] else 0
    if not dense_dim or not width:
        raise ValueError("Expected nonempty dense and multivector embeddings")
    shapes = {}
    for label, item in zip(LABELS, items, strict=True):
        dense, tokens = item["dense"], item["multivector"]
        if set(item) != {"dense", "multivector"} or len(dense) != dense_dim or not tokens:
            raise ValueError(f"Invalid embedding heads for {label}")
        if any(len(row) != width for row in tokens):
            raise ValueError(f"Inconsistent multivector width for {label}")
        if not all(math.isfinite(value) for value in dense) or not all(
            math.isfinite(value) for row in tokens for value in row
        ):
            raise ValueError(f"Non-finite embeddings for {label}")
        shapes[label] = {"dense": [dense_dim], "multivector": [len(tokens), width]}
    baseline, changed = items[2:]
    if len(baseline["multivector"]) != len(changed["multivector"]):
        raise ValueError("Same-resolution image variants should retain the same token count")
    if len(baseline["multivector"]) <= len(items[0]["multivector"]):
        raise ValueError("Image input did not add visual-token rows")
    dense_delta = max(abs(a - b) for a, b in zip(baseline["dense"], changed["dense"], strict=True))
    token_delta = max(
        abs(a - b)
        for left, right in zip(baseline["multivector"], changed["multivector"], strict=True)
        for a, b in zip(left, right, strict=True)
    )
    if dense_delta <= MIN_IMAGE_DELTA or token_delta <= MIN_IMAGE_DELTA:
        raise ValueError("Changing image pixels did not change both heads; check the visual path")
    return {
        "shapes": shapes,
        "image_change_max_abs_delta": {"dense": dense_delta, "multivector": token_delta},
    }
