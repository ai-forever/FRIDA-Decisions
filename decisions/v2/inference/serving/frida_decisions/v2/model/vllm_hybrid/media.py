"""A request's images beside its ids: from the IO processor to the pooler.

The ids carry the image tokens (`rows`), and these three tensors carry the rest: the
image processor's `pixel_values` and `image_grid_thw`, and the state's M-RoPE rows.
They travel as plain bytes in `PoolingParams.extra_kwargs`, which vLLM's msgpack moves
from the API server to the engine with the request; a tensor there would not come back
as one (its decoder knows tensors only in typed fields). So the image processor runs in
the API server, once per request, and the vision tower in the engine, once per request
in a pass (`causal_flat._Images`).

No vLLM in this file.
"""
from __future__ import annotations

import base64
import binascii
import io

import torch

KEY = "frida_hybrid_media"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    return t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()


def vision_pixel_dtype(folder, spec):
    """Resolve the loaded Qwen tower's input cast from the stored weights."""
    import json
    from safetensors import safe_open

    if "weights" not in spec or "vision" not in spec:
        return None
    model_type = json.loads((folder / "base" / "config.json").read_text())["model_type"]
    if model_type not in {"qwen3_5", "qwen3_5_moe"}:
        return None
    buffers = {"visual.rotary_pos_emb.inv_freq", "visual.rotary_pos_emb.original_inv_freq"}
    seen = set()
    for filename in spec["weights"]["files"]:
        with safe_open(str(folder / filename), framework="pt") as tensors:
            for key in tensors.keys():
                if key.startswith("visual.") and key not in buffers:
                    if tensors.get_slice(key).get_dtype() != "BF16":
                        return None
                    seen.add(key)
    return torch.bfloat16 if "visual.patch_embed.proj.weight" in seen else None


def rgb8_normalization_table(processor):
    """The installed still-image processor's exact BF16 result for each byte value."""
    from transformers import __version__ as transformers_version

    if (transformers_version != "5.19.0" or torch.__version__.split("+")[0] != "2.13.0"
            or type(processor).__module__ != "transformers.models.qwen2_vl.image_processing_qwen2_vl"
            or type(processor).__name__ != "Qwen2VLImageProcessor"
            or processor.temporal_patch_size != 2 or processor.patch_size != 16
            or not processor.do_rescale or not processor.do_normalize):
        return None
    ramp = torch.arange(256, dtype=torch.uint8).view(1, 1, 16, 16).expand(1, 3, 16, 16)
    normalized = processor.rescale_and_normalize(ramp, True, processor.rescale_factor,
                                                True, processor.image_mean, processor.image_std)
    bits = normalized.to(torch.bfloat16).contiguous().view(torch.int16)[0]
    if not torch.equal(bits[0], bits[1]) or not torch.equal(bits[0], bits[2]):
        return None
    return bits[0].numpy().tobytes()


def restore_rgb8(pixels, table):
    """Expand a single still frame and recover the tower's input BF16 bit patterns."""
    import numpy as np

    if len(table) != 512 or len(pixels) % (3 * 256):
        raise ValueError("Invalid Qwen RGB patch payload")
    src = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 3, 256)
    lut = np.frombuffer(table, dtype=np.uint16)
    first = lut[src]
    result = np.empty((len(src), 3, 2, 256), dtype=np.uint16)
    result[:, :, 0] = first
    result[:, :, 1] = first
    return torch.from_numpy(result.reshape(-1, 1536)).view(torch.bfloat16)


def to_params(plan, pixel_dtype=None) -> dict:
    """`extra_kwargs` of a plan with images."""
    mm, pos = plan.mm, plan.root.pos
    pixels = mm["pixel_values"]
    if "wire_rgb8_table" in mm:
        if pixel_dtype != torch.bfloat16 or pixels.dtype != torch.uint8:
            raise ValueError("The RGB byte wire requires a verified BF16 tower")
        return {KEY: {"pixels": _bytes(pixels), "rgb8_table": mm["wire_rgb8_table"],
                      "grid": mm["image_grid_thw"].tolist(), "positions": _bytes(pos.to(torch.int64))}}
    if pixel_dtype is not None:
        # Qwen casts these normalized pixels to the visual weights' dtype before its first op.
        pixels = pixels.to(pixel_dtype)
    name = next(k for k, v in DTYPES.items() if v == pixels.dtype)
    return {KEY: {"pixels": _bytes(pixels), "pixel_shape": list(pixels.shape), "pixel_dtype": name,
                  "grid": mm["image_grid_thw"].tolist(), "positions": _bytes(pos.to(torch.int64))}}


def from_params(extra: dict | None, *, defer_rgb8=False) -> dict | None:
    """The media `rows.decode` takes, or None for a request without images."""
    media = (extra or {}).get(KEY)
    if media is None:
        return None
    table = None
    if "rgb8_table" in media and defer_rgb8:
        if len(media["rgb8_table"]) != 512 or len(media["pixels"]) % 768:
            raise ValueError("Invalid Qwen RGB patch payload")
        pixels = torch.frombuffer(bytearray(media["pixels"]), dtype=torch.uint8).view(-1, 768)
        table = torch.frombuffer(bytearray(media["rgb8_table"]), dtype=torch.int16)
    elif "rgb8_table" in media:
        pixels = restore_rgb8(media["pixels"], media["rgb8_table"])
    else:
        kind = DTYPES[media["pixel_dtype"]]
        raw = torch.frombuffer(bytearray(media["pixels"]), dtype=torch.uint8)
        pixels = raw.view(kind).view(media["pixel_shape"])
    positions = torch.frombuffer(bytearray(media["positions"]), dtype=torch.int64).view(3, -1)
    result = {"pixel_values": pixels, "image_grid_thw": torch.tensor(media["grid"], dtype=torch.long),
              "positions": positions}
    if table is not None:
        result["rgb8_table"] = table
    return result


def decode_image(text: str):
    """A base64 image, bare or as a data URL, as an RGB PIL image."""
    from PIL import Image

    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"an image is not base64: {error}") from None
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")
