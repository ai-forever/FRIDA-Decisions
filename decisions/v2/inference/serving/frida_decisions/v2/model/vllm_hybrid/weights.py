"""Load standalone FRIDA-Decisions weights, buffers and image processing configuration."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


BASE = "base"
WEIGHTS = "weights"
SHARD_BYTES = 4 << 30
IMAGE_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def tensors_of(encoder) -> dict:
    """Every parameter and buffer, persistent or not, under its name in the encoder."""
    out = {name: t.detach() for name, t in encoder.named_parameters()}
    out.update((name, t.detach()) for name, t in encoder.named_buffers())
    return out




def _put(root, name: str, tensor) -> None:
    import torch

    owner, _, leaf = name.rpartition(".")
    module = root.get_submodule(owner)
    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in module._buffers:
        module._buffers[leaf] = tensor            # keeps the buffer's persistence
    elif leaf.endswith("_scale") and leaf[:-len("_scale")] in module._parameters:
        module.register_buffer(leaf, tensor)      # an FP8 expert matrix's block scales
    else:
        raise KeyError(f"{name}: the model has no such parameter or buffer")


def load(folder: Path, device, dtype=None):
    """The encoder a serving folder holds, its tensors read straight to `device`."""
    import torch
    from safetensors import safe_open
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    from frida_decisions.v2.model.causal import CausalScalarEncoder, load_config
    from frida_decisions.v2.model.render.causal_scalar import CausalScalarConfig
    from frida_decisions.v2.model.special_rows import SpecialRows

    folder = Path(folder)
    spec = json.loads((folder / "frida_hybrid.json").read_text(encoding="utf-8"))["weights"]
    fmt, base = spec["format"], str(folder / BASE)
    dtype = dtype or torch.bfloat16
    hf = load_config(base)
    with torch.device("meta"):
        backbone = AutoModel.from_config(hf, dtype=dtype, attn_implementation="sdpa")
    vision = fmt.get("vision")
    encoder = CausalScalarEncoder(base=base, dtype=dtype, config=CausalScalarConfig.from_dict(fmt["causal_scalar"]),
                                  executor=spec["executor"], backbone=backbone,
                                  tokenizer=AutoTokenizer.from_pretrained(base),
                                  drop=fmt.get("drop_layers", ()), vision=bool(vision),
                                  image_processor=AutoImageProcessor.from_pretrained(base) if vision else None,
                                  image_max_pixels=(vision or {}).get("max_pixels"))
    if fmt.get("train_tokens"):
        table = encoder.backbone.get_input_embeddings()
        encoder.backbone.set_input_embeddings(SpecialRows(table, fmt["train_tokens"], spec["special_rows_scale"]))
    read = set()
    for file in spec["files"]:
        with safe_open(str(folder / file), "pt", device=str(device)) as f:
            for name in f.keys():
                _put(encoder, name, f.get_tensor(name))
                read.add(name)
    left = [name for name, t in [*encoder.named_parameters(remove_duplicate=False),
                                 *encoder.named_buffers(remove_duplicate=False)] if t.is_meta]
    if left:
        raise ValueError(f"{folder}: {len(left)} tensors not in the folder, e.g. {left[:3]}")
    missing = set(tensors_of(encoder)) - read
    if missing:
        raise ValueError(f"{folder}: {len(missing)} tensors kept from the skeleton, e.g. {sorted(missing)[:3]}")
    return encoder.to(device).eval()


# ------------------------------------------------------------------- the job
















