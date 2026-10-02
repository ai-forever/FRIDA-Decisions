"""Merge a LoRA adapter into FRIDA and save a self-contained model folder.

    python tools/export_model.py --checkpoint path/to/adapter --out _export/FRIDA-Decisions

Input: a folder with `adapter_model.safetensors` + `adapter_config.json`
(LoRA on the encoder's attention projections) and `head.pt` (the linear head).

Output folder:

    config.json               T5 encoder config (dropout off)
    model.safetensors         merged encoder weights (bfloat16 by default)
    head.safetensors          decision head, float32: `weight` (1, d_model), `bias` (1,)
    tokenizer.json, tokenizer_config.json, ...
    decisions_config.json     packing limits, instruction suffixes, bucket settings

No peft needed, here or at inference: the merge is `W += (B @ A) * alpha / r`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frida_decisions.config import DecisionsConfig  # noqa: E402
from frida_decisions.constants import HEAD_WEIGHTS_FILE  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


def merge_lora_(model: torch.nn.Module, adapter_dir: str | Path) -> int:
    """Merge the adapter into `model` in place (in the model's dtype). Returns the
    number of merged weight matrices."""
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("fan_in_fan_out"):
        raise ValueError("only plain LoRA adapters are supported")
    scale = cfg["lora_alpha"] / cfg["r"]
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(model.named_parameters())
    merged = 0
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        # "base_model.model.<module path>.lora_A.weight" -> "<module path>.weight"
        module = key.split(".lora_A.")[0].removeprefix("base_model.model.")
        weight = params[module + ".weight"]
        with torch.no_grad():
            delta = (b.float() @ a.float()) * scale
            weight.add_(delta.to(weight.dtype))
        merged += 1
    if merged == 0:
        raise ValueError(f"no LoRA weights found in {adapter_dir}")
    return merged


def load_merged(checkpoint: str | Path, base: str = "ai-forever/FRIDA"):
    """Base encoder in float32 with the adapter merged, plus the head state dict."""
    from transformers import T5EncoderModel

    config = T5EncoderModel.config_class.from_pretrained(base)
    config.dropout_rate = 0.0
    model = T5EncoderModel.from_pretrained(base, config=config).float().eval()
    merged = merge_lora_(model, checkpoint)
    head = torch.load(Path(checkpoint) / "head.pt", map_location="cpu")
    return model, {"weight": head["weight"].float().contiguous(),
                   "bias": head["bias"].float().contiguous()}, merged


def export(checkpoint: str, out: str, base: str = "ai-forever/FRIDA", dtype: str = "bf16") -> Path:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, head, merged = load_merged(checkpoint, base)
    print(f"merged {merged} LoRA matrices")

    model = model.to(DTYPES[dtype])
    model.config.dropout_rate = 0.0
    model.config.name_or_path = ""
    model.save_pretrained(str(out_dir), safe_serialization=True)
    save_file(head, str(out_dir / HEAD_WEIGHTS_FILE))

    tokenizer = AutoTokenizer.from_pretrained(base)
    # Every segment is truncated by the packer; this only stops the tokenizer
    # from warning about packed rows longer than the base model's label.
    tokenizer.model_max_length = 1024
    tokenizer.save_pretrained(str(out_dir))

    cfg = DecisionsConfig(num_buckets=model.config.relative_attention_num_buckets,
                          max_distance=model.config.relative_attention_max_distance,
                          eos_token_id=model.config.eos_token_id,
                          pad_token_id=model.config.pad_token_id,
                          tokenizer_max_length=tokenizer.model_max_length)
    cfg.save(out_dir)
    _scrub_json(out_dir)
    print("wrote", out_dir)
    return out_dir


def _scrub_json(folder: Path) -> None:
    """Drop local-path fields that tooling writes into JSON configs."""
    for path in folder.glob("*.json"):
        if path.name == "tokenizer.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for key in ("_name_or_path", "name_or_path", "local_files_only", "is_local"):
            if key in data:
                data.pop(key)
                changed = True
        if changed:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="adapter folder (LoRA + head.pt)")
    parser.add_argument("--out", required=True, help="output model folder")
    parser.add_argument("--base", default="ai-forever/FRIDA", help="base encoder (repo id or path)")
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    args = parser.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    export(args.checkpoint, args.out, args.base, args.dtype)


if __name__ == "__main__":
    main()
