"""vLLM plugin entry points for FRIDA-Decisions V2."""
from __future__ import annotations

import json
from pathlib import Path

ARCH = "FridaHybridModel"
PLUGIN = "frida_decisions_v2"
SPEC = "frida_hybrid.json"
VISION = "vision:"                 # a variant read with its vision tower: `vision:<experiment>`
# What vLLM may read from a base's snapshot; the weights stay where the encoder loads them.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
                   "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")


def register() -> None:
    from vllm import ModelRegistry

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, f"{__name__}.model:{ARCH}")


def io_processor() -> str:
    return f"{__name__}.server.HybridIO"




