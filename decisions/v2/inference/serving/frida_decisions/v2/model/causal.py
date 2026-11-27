"""A causal LM (Qwen3 family) with one scalar per option.

Same head (`w.h + b` on the option's closing separator), same visibility
(option sees state, its question and itself), same `ScoringPlan` / `group`, so
the trainer's losses apply unchanged; only the backbone differs. The facade
mirrors `DecisionEncoder` where it can:

    model = CausalScalarEncoder("Qwen/Qwen3-0.6B")
    batch = model.pack([compile_request(r) for r in requests])    # or Compiled
    margins = model(batch.to(device)); batch.group                 # as Packed.group

    params = model.add_lora(rank=16, ff=True, train_tokens=model.renderer.separator_ids)
    model.save(out); CausalScalarEncoder.from_checkpoint(out)

The backbone is the HF *base* model (`AutoModel`: `last_hidden_state` after
the final norm); no LM head is loaded. Executors: `per_option` (default, any
attention kernel), `tree` (shared prefix with a 4D mask; the reference) or
`prefix` (the tree level by level over a KV cache: every segment computed once)
or `packed` (the tree's rows of several requests in fixed-length sequences,
with a FlexAttention block mask).
Training and job wiring are not here.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from frida_decisions.v2.model.exec.causal import (CausalBatch, CausalPerOptionExecutor,
                                                   CausalPrefixExecutor, CausalTreeExecutor)
from frida_decisions.v2.model.exec.causal_flat import (CausalFlatEagerExecutor, CausalFlatExecutor,
                                                        CausalFlatFp8Executor, CausalFlatPaddedExecutor)
from frida_decisions.v2.model.exec.causal_packed import CausalPackedExecutor
from frida_decisions.v2.model.head import LinearReadoutHead, scalar_head
from frida_decisions.v2.model.render.causal_scalar import (CAUSAL_SCALAR, STATE_FORMATS, VERSIONS,
                                                            CausalScalarConfig,
                                                            CausalScalarRenderer)
from frida_decisions.v2.model.special_rows import SpecialRows
from frida_decisions.v2.protocol import Compiled, Task

EXECUTORS = {"per_option": CausalPerOptionExecutor, "tree": CausalTreeExecutor,
             "prefix": CausalPrefixExecutor, "packed": CausalPackedExecutor,
             "flat": CausalFlatExecutor, "flat_padded": CausalFlatPaddedExecutor,
             "flat_eager": CausalFlatEagerExecutor, "flat_fp8": CausalFlatFp8Executor}
# LoRA targets of the Qwen / Llama family.
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
# Other token mixers, targeted where the backbone has them: DeepSeek-V3's MLA
# (GigaChat3) and Qwen3.5's Gated DeltaNet (the linear-attention layers).
MLA = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
LINEAR_ATTENTION = ["in_proj_qkv", "in_proj_z", "out_proj"]
EXPERTS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
# A fully fine-tuned backbone (`full_finetune`): its weights in bf16 beside the
# format file, in place of an adapter.
FULL_FILE = "backbone.safetensors"
# The small backbone tensors a LoRA run also trains (`unfreeze_small`), saved
# as fp32 offsets from the base beside the adapter. "Small": a frozen tensor of
# at most SMALL_MAX elements outside the adapted layers -- norms, Gated
# DeltaNet's conv1d / in_proj_a / in_proj_b / A_log / dt_bias, an MoE router;
# never an embedding table or fused experts.
SMALL_FILE = "small.safetensors"
SMALL_MAX = 1 << 22
PEFT_PREFIX = "base_model.model."




def recurrent(config) -> bool:
    """A hybrid backbone with recurrent (linear-attention) layers. Those read
    the sequence in order and ignore an attention mask, so an option needs a
    sequence of its own: `per_option`, or `prefix`, which forks the recurrent
    state at the end of the shared prefix. Under `tree` / `packed` an option
    would see its siblings. A vision-language checkpoint (Qwen3.5) keeps them
    in its `text_config`."""
    text = getattr(config, "text_config", None)
    return any("linear_attention" in (getattr(c, "layer_types", None) or ())
               for c in (config, text) if c is not None)


def load_config(base: str):
    """The backbone's config. transformers 5 validates field types, and some
    published configs carry an int where a float is declared (GigaChat3:
    `routed_scaling_factor: 1`); such fields are cast and the config rebuilt."""
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(base)
    except Exception as error:                  # transformers' strict-dataclass validation
        import typing

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.utils import cached_file

        raw = json.loads(Path(cached_file(base, "config.json")).read_text(encoding="utf-8"))
        cls = CONFIG_MAPPING[raw["model_type"]]
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = getattr(cls, "__annotations__", {})
        cast = []
        for field, value in list(raw.items()):
            declared = str(hints.get(field, ""))
            if isinstance(value, int) and not isinstance(value, bool) and \
                    ("float" in declared or hints.get(field) is float) and "int" not in declared:
                raw[field] = float(value)
                cast.append(field)
        if not cast:
            raise
        try:
            return cls(**raw)
        except Exception:
            raise error from None


def layer_stack(backbone: nn.Module) -> nn.Module:
    """The module that owns the decoder `layers` (through a PEFT wrapper too)."""
    inner = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    if not isinstance(getattr(inner, "layers", None), nn.ModuleList):
        raise ValueError(f"{type(inner).__name__} has no `layers` list to prune")
    return inner


def drop_layers(backbone: nn.Module, indices) -> list[int]:
    """Remove decoder layers (indices into the current stack) for real: the
    layer list, `config.layer_types`, `num_hidden_layers` and every
    sub-module's `layer_idx` (cache slots) are rewritten, so the result is an
    ordinary smaller model of the same class, not identity-masked layers.
    Returns the kept original indices."""
    stack = layer_stack(backbone)
    n = len(stack.layers)
    drop = sorted({int(i) for i in indices})
    if any(i < 0 or i >= n for i in drop):
        raise ValueError(f"layers to drop {drop} are not all in 0..{n - 1}")
    if len(drop) >= n:
        raise ValueError("cannot drop every layer")
    keep = [i for i in range(n) if i not in set(drop)]
    stack.layers = nn.ModuleList(stack.layers[i] for i in keep)
    config = stack.config
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[i] for i in keep]
    config.num_hidden_layers = len(keep)
    for new, layer in enumerate(stack.layers):
        for sub in layer.modules():
            if hasattr(sub, "layer_idx"):
                sub.layer_idx = new
    return keep


class CausalScalarEncoder(nn.Module):

    def __init__(self, base: str = "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
                 config: CausalScalarConfig | None = None, executor: str = "per_option",
                 attn_implementation: str = "sdpa", backbone: nn.Module | None = None,
                 tokenizer=None, drop: Sequence[int] = (), vision: bool = False,
                 image_processor=None, image_max_pixels: int | None = None):
        """`backbone` / `tokenizer` may be passed in (tests build a tiny random one).
        `drop`: decoder layers removed from the base model before anything else
        (layer-drop ablation); recorded in the format file, so a checkpoint
        reloads with the same layers. `vision`: keep a VL checkpoint's vision
        tower and image processor, so a request's state may carry images
        (`pack(requests, images=...)`, model/vision.py); the text path is
        unchanged, and adapters stay on the text model. `image_max_pixels` caps
        an image's area (the processor resizes, keeping the aspect ratio);
        None keeps the checkpoint's bounds."""
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.base = base
        self.dropped = sorted({int(i) for i in drop})
        self.full = False                   # `full_finetune`: every backbone weight trains
        self.small = {}                     # `unfreeze_small`: name -> (tensor, its base copy)
        self.neftune_alpha = 0.0
        self._neftune_hook = None
        hf_config = load_config(base) if backbone is None else None
        self.tokenizer = tokenizer if tokenizer is not None else \
            AutoTokenizer.from_pretrained(base, config=hf_config)
        if backbone is None:
            backbone = AutoModel.from_pretrained(base, config=hf_config, dtype=dtype,
                                                 attn_implementation=attn_implementation)
        # A vision-language checkpoint (Qwen3.5) loads with its vision tower; the
        # text model is the backbone, and the tower is kept only with `vision`.
        vl = backbone if hasattr(backbone, "language_model") else None
        if vl is not None:
            backbone = vl.language_model
        self.vision = None
        if vision:
            from frida_decisions.v2.model.vision import VisionInputs

            if vl is None:
                raise ValueError(f"{base} has no vision tower")
            if image_processor is None:
                from transformers import AutoProcessor

                image_processor = AutoProcessor.from_pretrained(base).image_processor
            vl.visual.requires_grad_(False)
            from frida_decisions.v2.model.vision import linear_patch_embed

            linear_patch_embed(vl.visual.patch_embed)   # not the slow Conv3d path
            self.visual = vl.visual              # a submodule: moves with the encoder
            self.vision = VisionInputs(vl, image_processor, max_pixels=image_max_pixels)
        if self.dropped:
            drop_layers(backbone, self.dropped)
        self.backbone = backbone
        self.config = config or CausalScalarConfig()
        self.renderer = CausalScalarRenderer(self.tokenizer, self.config)
        self.renderer.media = self.vision
        self.head = scalar_head(backbone.config.hidden_size)
        if executor not in EXECUTORS:
            raise ValueError(f"executor {executor!r} is not one of {sorted(EXECUTORS)}")
        self.executor_kind = executor
        self._check_executor()
        # Set by `from_checkpoint` when states are rendered in another format
        # than the one trained on.
        self.render_note: str | None = None

    # ------------------------------------------------------------ the layers
    def _check_executor(self):
        """A hybrid backbone (recurrent layers) is exact only under per_option,
        prefix and flat. Checked wherever an executor is built: `executor_kind` is
        switched after construction by tests, steptime and benches."""
        if recurrent(self.backbone.config) and self.executor_kind not in ("per_option", "prefix", "flat", "flat_padded", "flat_eager", "flat_fp8"):
            raise ValueError(f"{self.base} has recurrent (linear-attention) layers, which "
                             f"ignore the {self.executor_kind} executor's mask: options would "
                             "see each other. Use per_option, prefix or flat.")

    @property
    def executor(self):
        self._check_executor()
        return EXECUTORS[self.executor_kind](self)

    def warm_up(self, after_pass=None) -> None:
        """Compile what a first batch would otherwise pay for on a GPU: the kernels of short and
        long rows, forks and several requests.
        `after_pass(i)` is called after each of its passes (the plugin logs memory there)."""
        from frida_decisions.v2.protocol import compile_request, parse_request
        questions = {f"q{q}": {"type": "choice", "instructions": f"Question {q}?",
                               "criteria": {f"k{k}": "option " * (1 + 40 * k) for k in range(8)}}
                     for q in range(3)}
        sized = [compile_request(parse_request({"state": "word " * n, "questions": questions}))
                 for n in (16, 900, 2000, 6000)]
        device = next(self.parameters()).device
        with torch.inference_mode():
            for i, requests in enumerate([[r] for r in sized] + [sized]):
                self(self.pack(requests).to(device))
                if after_pass is not None:
                    after_pass(i)

    @property
    def readout_head(self) -> LinearReadoutHead:
        return LinearReadoutHead(self.head)

    @property
    def pad_token_id(self) -> int:
        pad = self.tokenizer.pad_token_id
        return int(pad if pad is not None else self.tokenizer.eos_token_id)

    def pack(self, requests: Sequence[Sequence[Task] | Compiled], images=None) -> CausalBatch:
        """`images`: per request, a list of paths / PIL images (or None) for its
        state; needs `vision=True` and the per_option or flat executor."""
        images = images or [None] * len(requests)
        return self.executor.pack([self.renderer.render(r, imgs)
                                   for r, imgs in zip(requests, images)])

    def forward(self, batch: CausalBatch) -> torch.Tensor:
        """One margin per (question, option), in packing order."""
        return self.executor.forward(batch)

    # ---------------------------------------------------------------- adapters








    @property
    def special_rows(self) -> SpecialRows | None:
        table = self.backbone.get_input_embeddings()
        return table if isinstance(table, SpecialRows) else None

    # ---------------------------------------------------------- checkpoints







