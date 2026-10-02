"""PyTorch backend: GPU or CPU, packed rows plus an exact state cache."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .base import BaseJudge, resolve_model_dir
from .constants import DEFAULT_REPO_ID, HEAD_WEIGHTS_FILE
from .modeling import DecisionEncoder, StateCache, as_tensor
from .packing import (TokenizedRequest, build_rows, count_packed_rows, pack_queries, pack_rows,
                      state_buckets)

_TORCH_FILES = ["*.json", "model.safetensors", HEAD_WEIGHTS_FILE]


def load_head(folder: Path, d_model: int) -> nn.Linear:
    """The decision head from `head.safetensors` (float32 `weight` (1, d) and `bias` (1,))."""
    from safetensors.torch import load_file

    weights = load_file(str(Path(folder) / HEAD_WEIGHTS_FILE))
    head = nn.Linear(d_model, 1, dtype=torch.float32)
    head.load_state_dict({"weight": weights["weight"].float(), "bias": weights["bias"].float()})
    return head


class Judge(BaseJudge):
    """Score requests with PyTorch.

        judge = Judge.from_pretrained("ai-forever/FRIDA-Decisions", device="cuda")
        judge({"state": "...", "questions": {...}})["answers"]

    One request is packed into as few rows as the limits allow and scored in
    one forward. When a request would need several rows, or its state was seen
    recently, the state is encoded once and its per-layer keys/values are
    reused (exact, not an approximation).
    """

    backend = "torch"

    def __init__(self, folder: Path, encoder: DecisionEncoder, device: torch.device,
                 state_max: int | None = None, state_cache_mb: int = 512,
                 rows_per_forward: int | None = None):
        super().__init__(folder, state_max)
        self.model = encoder.eval()
        self.device = torch.device(device)
        self.dtype = encoder.embed.weight.dtype
        self.state_cache = StateCache(state_cache_mb * 2**20) if state_cache_mb else None
        self.rows_per_forward = rows_per_forward

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = DEFAULT_REPO_ID, device: str | None = None,
                        dtype: torch.dtype | None = None, state_max: int = 384,
                        state_cache_mb: int = 512, rows_per_forward: int | None = None,
                        revision: str | None = None) -> "Judge":
        """Load an exported model folder or Hugging Face repo.

        device: "cuda", "cpu" or any torch device; default cuda when available.
        dtype: default bfloat16 on GPU and float32 on CPU.
        state_max: state tokens kept (the rest is cut). Up to 512 is the trained
            range; 384 is the default.
        state_cache_mb: memory budget of the state K/V cache, 0 turns it off.
        rows_per_forward: cap on packed rows per encoder call (default: all at once).
        """
        from transformers import T5EncoderModel

        folder = resolve_model_dir(path_or_repo, _TORCH_FILES, revision)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        t5 = T5EncoderModel.from_pretrained(str(folder))
        t5 = t5.to(device=device, dtype=dtype).eval()
        head = load_head(folder, t5.config.d_model).to(device)
        return cls(folder, DecisionEncoder(t5, head), device, state_max, state_cache_mb,
                   rows_per_forward)

    # ------------------------------------------------------------------ scoring
    def use_state_cache(self, requests: list[TokenizedRequest]) -> bool:
        """The cached path wins when the state is already cached, or when packing
        would split the request into several rows (each repeating the state).
        On a first, single-row request it would only add a second pass."""
        if self.state_cache is None or len(requests) != 1:
            return False
        req = requests[0]
        return tuple(req.state) in self.state_cache or count_packed_rows(req, self.config) > 1

    @torch.inference_mode()
    def _score(self, requests: list[TokenizedRequest]):
        if self.use_state_cache(requests):
            margins, usage = self._score_cached(requests[0])
            return [margins], usage
        return self._score_packed(requests)

    def _score_packed(self, requests: list[TokenizedRequest]):
        rows, per_request = build_rows(requests, self.config)
        step = self.rows_per_forward or len(rows)
        margins: list[float] = []
        tokens = 0
        for i in range(0, len(rows), step):
            batch = pack_rows(rows[i:i + step], self.config)
            margins += self._forward_packed(batch)
            tokens += sum(batch.lengths)
        out, start = [], 0
        for n in per_request:
            out.append(margins[start:start + n])
            start += n
        return out, {"rows": len(rows), "encoder_tokens": tokens, "state_cache": "off"}

    def _forward_packed(self, batch) -> list[float]:
        dev = self.device
        hidden = self.model(as_tensor(batch.input_ids, dev), as_tensor(batch.buckets, dev),
                            as_tensor(batch.allowed, dev))
        r = batch.readout
        return self.model.margins(hidden, as_tensor(r.row, dev), as_tensor(r.col, dev),
                                  as_tensor(r.slot, dev), r.count).float().cpu().tolist()

    def _score_cached(self, req: TokenizedRequest):
        dev = self.device
        key = tuple(req.state)
        hit = self.state_cache.get(key)
        state_cost = 0
        if hit is None:
            n = len(req.state)
            buckets, allowed = state_buckets(n, self.config)
            ids = torch.full((1, buckets.shape[0]), self.config.pad_token_id, dtype=torch.long)
            ids[0, :n] = torch.tensor(req.state, dtype=torch.long)
            ks, vs = self.model.encode_state(ids.to(dev), as_tensor(buckets[None], dev),
                                             as_tensor(allowed[None], dev), n)
            self.state_cache.put(key, ks, vs)
            state_cost = n
        else:
            ks, vs = hit
        q = pack_queries(req, self.config)
        hidden = self.model.forward_cached(as_tensor(q.input_ids, dev), as_tensor(q.buckets, dev),
                                           as_tensor(q.allowed, dev), ks, vs)
        r = q.readout
        margins = self.model.margins(hidden, as_tensor(r.row, dev), as_tensor(r.col, dev),
                                     as_tensor(r.slot, dev), r.count).float().cpu().tolist()
        tokens = state_cost + sum(len(row.ids) for row in q.rows)
        return margins, {"rows": len(q.rows), "encoder_tokens": tokens,
                         "state_cache": "miss" if hit is None else "hit"}

    # ------------------------------------------------------------------ explicit paths
    @torch.inference_mode()
    def margins_packed(self, request: dict) -> list[float]:
        """Margins through the packed path only (no cache). For tests and comparisons."""
        _, _, tok = self.compile(request)
        return self._score_packed([tok])[0][0]

    @torch.inference_mode()
    def margins_cached(self, request: dict) -> list[float]:
        """Margins through the state-cache path only. For tests and comparisons."""
        if self.state_cache is None:
            self.state_cache = StateCache()
        _, _, tok = self.compile(request)
        return self._score_cached(tok)[0]
