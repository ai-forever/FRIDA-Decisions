"""Images in the state of a vision-language causal backbone (Qwen3.5).

The decisions stay as they are: question, options and readout are text, and
only the STATE segment carries pictures. A request's images are placed at the
start of the state, before its text, each as the model's own block
`<|vision_start|> <|image_pad|> x n <|vision_end|>` (n = merged patches), the
layout bench/vlm_probe.py uses. Everything that depends on the model is taken
from it rather than re-derived here:

* pixels and the patch grid: the checkpoint's image processor;
* positions: the model's `get_rope_index` on the state alone (M-RoPE, three
  rows: time, height, width); text after the state continues from its max + 1,
  as the model's own text after an image does (docs/design/BACKBONE_AND_MODALITIES.md);
* image embeddings: the model's `get_image_features`, put in the place of the
  `<|image_pad|>` tokens, as `Qwen3_5Model.forward` does.

The per_option executor reads such plans (one plain causal sequence per option,
the layout the model was trained on), and flat, which gives the same margins with
the state once (`exec/causal_flat.py`); the others refuse them.
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_image(image):
    """A path (str / Path) or a PIL image, as an RGB PIL image."""
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as im:
            return im.convert("RGB")
    return image.convert("RGB")


def linear_patch_embed(patch_embed) -> bool:
    """Run a Qwen-VL patch embedding (a Conv3d whose stride equals its kernel,
    over inputs exactly one kernel in size) as the matrix product it is.

    On the job image, torch takes `slow_conv_dilated3d` for this bf16 Conv3d:
    ~13 s and ~3M element copies per 1-MP image, where the product takes
    milliseconds. Same weights, same result up to summation order. Returns
    whether the module was patched."""
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, torch.nn.Conv3d) or tuple(proj.stride) != tuple(proj.kernel_size):
        return False

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        weight = proj.weight
        flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
        return torch.nn.functional.linear(flat, weight.reshape(weight.shape[0], -1), proj.bias)

    patch_embed.forward = forward
    return True


def _stamp(path) -> tuple:
    try:
        st = Path(path).stat()
        return st.st_size, st.st_mtime_ns
    except OSError:
        return (None, None)


class VisionInputs:
    """The vision side of one VL checkpoint: what a renderer and an executor
    need to put images into a state. Holds the VL model by reference (not as a
    submodule: its vision tower is registered on the encoder, its text model is
    the encoder's backbone)."""

    def __init__(self, vl_model, image_processor, max_pixels: int | None = None):
        config = vl_model.config
        self.image_token_id = int(config.image_token_id)
        self.start_id = int(config.vision_start_token_id)
        self.end_id = int(config.vision_end_token_id)
        self.merge = int(config.vision_config.spatial_merge_size)
        self.image_processor = image_processor
        # None: the checkpoint's own bounds (Qwen3.5: 65,536-16,777,216 pixels, i.e.
        # native resolution up to 4096 x 4096, ~4K tokens for a 2048 x 2048 image).
        self.max_pixels = max_pixels
        self._vl = [vl_model]
        # The trainer renders a row more than once per step (cost, balance,
        # forward); the last few images by path are kept processed.
        self._cache: dict = {}

    @property
    def vl(self):
        return self._vl[0]

    def prepare(self, images) -> dict:
        """`pixel_values` and `image_grid_thw` of a request's images, in order."""
        key = None
        if all(isinstance(i, (str, Path)) for i in images):
            # The file's size and mtime too: an image replaced on disk while a
            # long-lived process runs is processed again, not served stale.
            key = tuple((str(i), *_stamp(i)) for i in images)
        if key is not None and key in self._cache:
            return self._cache[key]
        out = self._prepare(images)
        if key is not None:
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = out
        return out

    def _prepare(self, images) -> dict:
        extra = {}
        if self.max_pixels is not None:
            size = dict(getattr(self.image_processor, "size", None) or {})
            size["longest_edge"] = int(self.max_pixels)
            size["shortest_edge"] = min(int(size.get("shortest_edge", 0)), int(self.max_pixels))
            extra["size"] = size
        table = getattr(self, "wire_rgb8_table", None)
        if table is not None:
            extra.update(do_rescale=False, do_normalize=False, temporal_patch_size=1)
        out = self.image_processor(images=[load_image(i) for i in images], return_tensors="pt",
                                   **extra)
        mm = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}
        if table is not None:
            if mm["pixel_values"].dtype != torch.uint8 or mm["pixel_values"].shape[1] != 768:
                raise ValueError("Expected RGB byte patches from the serving image processor")
            mm["wire_rgb8_table"] = table
        return mm

    def block(self, grid) -> list[int]:
        """One image's tokens: start, one pad per merged patch, end."""
        n = int(torch.as_tensor(grid).prod()) // (self.merge ** 2)
        return [self.start_id, *[self.image_token_id] * n, self.end_id]

    def positions(self, ids: list[int], grid: torch.Tensor) -> torch.Tensor:
        """(3, len(ids)) M-RoPE positions of a state, from the model itself."""
        input_ids = torch.tensor([ids], dtype=torch.long)
        kinds = (input_ids == self.image_token_id).int()
        pos, _ = self.vl.get_rope_index(input_ids, mm_token_type_ids=kinds, image_grid_thw=grid)
        return pos[:, 0]

    def features(self, mm: dict) -> torch.Tensor:
        """(image tokens, hidden): the request's images through the vision tower.
        A frozen tower runs without autograd: nothing upstream of the splice
        trains, so its activations need not be kept."""
        frozen = not any(p.requires_grad for p in self.vl.visual.parameters())
        with torch.no_grad() if frozen else torch.enable_grad():
            visual = self.vl.visual
            host_grid = mm.get("host_grid")
            if (host_grid is not None and mm["pixel_values"].is_cuda and not visual.training
                    and visual.dtype == torch.bfloat16 and torch.cuda.get_device_capability(mm["pixel_values"].device) == (9, 0)
                    and type(visual).__name__ in ("Qwen3_5VisionModel", "Qwen3_5MoeVisionModel")
                    and getattr(visual.config, "_attn_implementation", None) == "sdpa"):
                try:
                    from transformers import __version__ as transformers_version
                    from transformers.vision_utils import get_vision_cu_seqlens
                except ImportError:
                    pass
                else:
                    # SDPA splits the batch by these lengths on the host in every vision block.
                    # The tower already returns merged features in the input image order.
                    if transformers_version == "5.19.0" and torch.__version__.split("+")[0] == "2.13.0":
                        boundaries = get_vision_cu_seqlens(host_grid)
                        return visual(mm["pixel_values"].to(visual.dtype), grid_thw=mm["image_grid_thw"],
                                      cu_seqlens=boundaries, return_dict=True).pooler_output
            out = self.vl.get_image_features(mm["pixel_values"], mm["image_grid_thw"],
                                             return_dict=True)
        return torch.cat(out.pooler_output, dim=0)
