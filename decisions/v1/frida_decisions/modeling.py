"""The encoder forward with a custom bias, plus the state key/value cache (torch).

The encoder is a stock `T5EncoderModel`. What differs from its own forward is
the attention bias: T5 computes it from `arange` positions, and here it comes
from the packed layout (restarting positions plus the block mask). Rather than
patching T5's internals, this module runs the same submodules in the same
order with our bias:

    x = embed(ids)
    for block:  x = x + Attention(LayerNorm(x), bias);  x = FeedForward(x)
    hidden = FinalLayerNorm(x)

T5 does not scale `q @ k^T`, hence `scale=1.0` below. The bias is built
contiguous, which is what lets SDPA pick a fused kernel on GPU.

The decision head is linear over the mean of a candidate's own token states.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DecisionEncoder(nn.Module):
    """A T5 encoder plus the linear decision head."""

    def __init__(self, t5_encoder: nn.Module, head: nn.Linear):
        super().__init__()
        self.t5 = t5_encoder                    # transformers.T5EncoderModel
        self.head = head                        # Linear(d_model, 1), kept in float32
        stack = t5_encoder.encoder
        self.embed = stack.embed_tokens
        self.blocks = stack.block
        self.final_norm = stack.final_layer_norm
        first = self.blocks[0].layer[0].SelfAttention
        self.bias_table = first.relative_attention_bias      # Embedding(num_buckets, heads)
        self.n_heads = first.n_heads
        self.d_kv = first.key_value_proj_dim

    # ------------------------------------------------------------- pieces
    def attention_bias(self, buckets: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """(B, Q, K) buckets + mask -> (B, heads, Q, K) additive bias."""
        bias = self.bias_table(buckets.long()).permute(0, 3, 1, 2)
        floor = torch.finfo(bias.dtype).min
        return bias.masked_fill(~allowed[:, None], floor).contiguous()

    def _qkv(self, sa, h):
        b, n, _ = h.shape
        shape = (b, n, self.n_heads, self.d_kv)
        q = sa.q(h).view(shape).transpose(1, 2)
        k = sa.k(h).view(shape).transpose(1, 2)
        v = sa.v(h).view(shape).transpose(1, 2)
        return q, k, v

    def _attend(self, sa, q, k, v, bias):
        b, _, n, _ = q.shape
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        return sa.o(out.transpose(1, 2).reshape(b, n, -1))

    # ------------------------------------------------------------- packed path
    def forward(self, input_ids: torch.Tensor, buckets: torch.Tensor,
                allowed: torch.Tensor) -> torch.Tensor:
        """Hidden states `(B, L, d_model)` of packed rows."""
        bias = self.attention_bias(buckets, allowed).to(self.embed.weight.dtype)
        x = self.embed(input_ids)
        for block in self.blocks:
            attn = block.layer[0]
            h = attn.layer_norm(x)
            q, k, v = self._qkv(attn.SelfAttention, h)
            x = x + self._attend(attn.SelfAttention, q, k, v, bias)
            x = block.layer[-1](x)              # T5LayerFF: norm, gated FFN, residual
        return self.final_norm(x)

    def margins(self, hidden: torch.Tensor, read_row, read_col, read_slot,
                count: int) -> torch.Tensor:
        """Mean over each candidate's tokens, then the head. Float32 throughout."""
        picked = hidden[read_row, read_col].float()
        pooled = torch.zeros(count, picked.shape[-1], device=picked.device)
        counts = torch.zeros(count, 1, device=picked.device)
        pooled.index_add_(0, read_slot, picked)
        counts.index_add_(0, read_slot, torch.ones_like(picked[:, :1]))
        return self.head(pooled / counts).squeeze(-1)

    # ------------------------------------------------------------- cached path
    def encode_state(self, ids: torch.Tensor, buckets: torch.Tensor, allowed: torch.Tensor,
                     n: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-layer keys/values of a state encoded on its own.

        The state attends only to itself in the packed layout, so its keys and
        values do not depend on the questions or options and can be reused.
        The last layer only needs its K/V: its output at state positions is
        never read, so its attention and FFN are skipped.
        """
        bias = self.attention_bias(buckets, allowed).to(self.embed.weight.dtype)
        x = self.embed(ids)
        ks, vs = [], []
        for i, block in enumerate(self.blocks):
            attn = block.layer[0]
            h = attn.layer_norm(x)
            q, k, v = self._qkv(attn.SelfAttention, h)
            ks.append(k[:, :, :n].contiguous())
            vs.append(v[:, :, :n].contiguous())
            if i == len(self.blocks) - 1:
                break
            x = x + self._attend(attn.SelfAttention, q, k, v, bias)
            x = block.layer[-1](x)
        return ks, vs

    def forward_cached(self, input_ids: torch.Tensor, buckets: torch.Tensor,
                       allowed: torch.Tensor, ks: list, vs: list) -> torch.Tensor:
        """Hidden states of question/option rows, with the cached state K/V in front."""
        bias = self.attention_bias(buckets, allowed).to(self.embed.weight.dtype)
        b = input_ids.shape[0]
        x = self.embed(input_ids)
        for i, block in enumerate(self.blocks):
            attn = block.layer[0]
            h = attn.layer_norm(x)
            q, k, v = self._qkv(attn.SelfAttention, h)
            k_all = torch.cat([ks[i].expand(b, -1, -1, -1), k], dim=2)
            v_all = torch.cat([vs[i].expand(b, -1, -1, -1), v], dim=2)
            x = x + self._attend(attn.SelfAttention, q, k_all, v_all, bias)
            x = block.layer[-1](x)
        return self.final_norm(x)


class TokenScores(nn.Module):
    """Export wrapper: per-token head scores `(B, L)`.

    The head is linear, so the mean of per-token scores over a candidate's
    span equals the head applied to the mean state. Folding it into the graph
    leaves the ONNX consumer a single average per candidate to compute.
    """

    def __init__(self, encoder: DecisionEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, buckets, allowed):
        hidden = self.encoder(input_ids, buckets, allowed).float()
        return self.encoder.head(hidden).squeeze(-1)


class StateCache:
    """LRU of per-layer state keys/values, bounded in bytes."""

    def __init__(self, max_bytes: int = 512 * 2**20):
        self.max_bytes = max_bytes
        self._items: OrderedDict = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def __contains__(self, key) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return item[0], item[1]

    def put(self, key, ks, vs) -> None:
        size = sum(t.numel() * t.element_size() for t in ks + vs)
        if size > self.max_bytes:
            return                  # a single state larger than the budget is not cached
        if key in self._items:
            self.bytes -= self._items.pop(key)[2]
        self._items[key] = (ks, vs, size)
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, (_, _, evicted) = self._items.popitem(last=False)
            self.bytes -= evicted

    def clear(self) -> None:
        self._items.clear()
        self.bytes = 0


def as_tensor(array: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)
