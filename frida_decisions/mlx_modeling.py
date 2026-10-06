"""Native encoder inference with FRIDA's supplied attention layout.

The layer structure follows the T5 example in ml-explore/mlx-examples,
https://github.com/ml-explore/mlx-examples/blob/main/t5/t5.py
(Copyright (c) 2023 Apple Inc., MIT License). Only the encoder is needed. Buckets and visibility come from packing.py.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def gelu_new(x):
    """Hugging Face's tanh GELU approximation."""
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) *
                                   (x + 0.044715 * mx.power(x, 3))))


class T5Norm(nn.Module):
    """T5 normalization: FP32 variance, then the parameter's precision."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        h = x.astype(mx.float32)
        h = h * mx.rsqrt(mx.mean(mx.square(h), axis=-1, keepdims=True) + self.eps)
        return self.weight * h.astype(self.weight.dtype)


class FridaMlxAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.d_kv = cfg.d_kv
        inner = cfg.num_heads * cfg.d_kv
        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)

    def attend(self, q, k, v, bias):
        # FRIDA/T5 uses unscaled attention and the supplied additive bias.
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=bias)

    def __call__(self, x, bias):
        b, n, _ = x.shape
        shape = (b, n, self.num_heads, self.d_kv)
        q, k, v = [p(x).reshape(shape).transpose(0, 2, 1, 3)
                   for p in (self.q, self.k, self.v)]
        out = self.attend(q, k, v, bias).transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.o(out)


class FridaMlxDenseActivation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wi_0 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wi_1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x):
        return self.wo(gelu_new(self.wi_0(x)) * self.wi_1(x))


class FridaMlxEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = FridaMlxAttention(cfg)
        self.ln1 = T5Norm(cfg.d_model, cfg.layer_norm_epsilon)
        self.ln2 = T5Norm(cfg.d_model, cfg.layer_norm_epsilon)
        self.dense = FridaMlxDenseActivation(cfg)

    def __call__(self, x, bias):
        x = x + self.attention(self.ln1(x), bias)
        return x + self.dense(self.ln2(x))


class FridaMlxDecisionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model_type != "t5" or cfg.feed_forward_proj != "gated-gelu":
            raise ValueError("MLX requires a T5 encoder with gated-gelu")
        if getattr(cfg, "dense_act_fn", "gelu_new") != "gelu_new":
            raise ValueError("MLX requires gelu_new")
        if getattr(cfg, "is_decoder", False):
            raise ValueError("MLX requires an encoder configuration")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [FridaMlxEncoderLayer(cfg) for _ in range(cfg.num_layers)]
        self.final_norm = T5Norm(cfg.d_model, cfg.layer_norm_epsilon)
        self.bias_table = nn.Embedding(cfg.relative_attention_num_buckets, cfg.num_heads)
        self.head = nn.Linear(cfg.d_model, 1)

    def attention_bias(self, buckets, allowed):
        bias = self.bias_table(buckets).transpose(0, 3, 1, 2)
        return mx.where(allowed[:, None], bias, mx.finfo(bias.dtype).min)

    def __call__(self, input_ids, buckets, allowed):
        bias = self.attention_bias(buckets, allowed)
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, bias)
        return self.final_norm(x)

    def encode_state(self, ids, buckets, allowed, n):
        """Encode independent state keys and values for every layer."""
        if n == 0:
            # MLX projection kernels require a nonempty sequence.
            empty = [mx.zeros((1, layer.attention.num_heads, 0, layer.attention.d_kv),
                              dtype=self.embed.weight.dtype) for layer in self.layers]
            return empty, list(empty)
        bias = self.attention_bias(buckets, allowed)
        x = self.embed(ids)
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            h = layer.ln1(x)
            b, width, _ = h.shape
            shape = (b, width, layer.attention.num_heads, layer.attention.d_kv)
            k, v = [p(h).reshape(shape).transpose(0, 2, 1, 3)
                    for p in (layer.attention.k, layer.attention.v)]
            # Copies retain only real state tokens, without padded storage.
            ks.append(mx.contiguous(k[:, :, :n]))
            vs.append(mx.contiguous(v[:, :, :n]))
            if i != len(self.layers) - 1:
                x = x + layer.attention(h, bias)
                x = x + layer.dense(layer.ln2(x))
        mx.eval(ks, vs)
        return ks, vs

    def forward_cached(self, input_ids, buckets, allowed, ks, vs):
        """Return row hidden states with independent state keys and values."""
        bias = self.attention_bias(buckets, allowed)
        x = self.embed(input_ids)
        b, n, _ = x.shape
        for i, layer in enumerate(self.layers):
            attn = layer.attention
            h = layer.ln1(x)
            shape = (b, n, attn.num_heads, attn.d_kv)
            q, k, v = [p(h).reshape(shape).transpose(0, 2, 1, 3)
                       for p in (attn.q, attn.k, attn.v)]
            k = mx.concatenate([mx.broadcast_to(ks[i], (b, *ks[i].shape[1:])), k], axis=2)
            v = mx.concatenate([mx.broadcast_to(vs[i], (b, *vs[i].shape[1:])), v], axis=2)
            out = attn.attend(q, k, v, bias).transpose(0, 2, 1, 3).reshape(b, n, -1)
            x = x + attn.o(out)
            x = x + layer.dense(layer.ln2(x))
        return self.final_norm(x)

    def pool(self, hidden, row, col, slot, count):
        picked = hidden[row, col].astype(mx.float32)
        pooled = mx.zeros((count, picked.shape[-1]), dtype=mx.float32).at[slot].add(picked)
        counts = mx.zeros((count, 1), dtype=mx.float32).at[slot].add(
            mx.ones((picked.shape[0], 1), dtype=mx.float32))
        return pooled / counts

    def margins(self, hidden, row, col, slot, count):
        return self.head(self.pool(hidden, row, col, slot, count)).squeeze(-1)

    def weight_mapping(self):
        mapping = {"shared.weight": "embed.weight",
                   "encoder.final_layer_norm.weight": "final_norm.weight",
                   "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight":
                       "bias_table.weight"}
        for i in range(len(self.layers)):
            source, target = f"encoder.block.{i}", f"layers.{i}"
            for projection in ("q", "k", "v", "o"):
                mapping[f"{source}.layer.0.SelfAttention.{projection}.weight"] = (
                    f"{target}.attention.{projection}.weight")
            for projection in ("wi_0", "wi_1", "wo"):
                mapping[f"{source}.layer.1.DenseReluDense.{projection}.weight"] = (
                    f"{target}.dense.{projection}.weight")
            mapping[f"{source}.layer.0.layer_norm.weight"] = f"{target}.ln1.weight"
            mapping[f"{source}.layer.1.layer_norm.weight"] = f"{target}.ln2.weight"
        return mapping

    def load_checkpoint(self, weights, head, dtype):
        if dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("dtype must be mlx.core.float32 or mlx.core.bfloat16")
        weights = dict(weights)
        # Some T5 exports retain an explicit copy of the tied embedding.
        alias = weights.pop("encoder.embed_tokens.weight", None)
        if "shared.weight" not in weights and alias is not None:
            weights["shared.weight"] = alias
        elif alias is not None:
            shared = weights["shared.weight"]
            if alias.shape != shared.shape or not mx.array_equal(alias, shared).item():
                raise ValueError("Tied encoder embeddings do not match shared.weight")
        mapping = self.weight_mapping()
        missing, extra = set(mapping) - set(weights), set(weights) - set(mapping)
        if missing or extra:
            raise ValueError(f"Encoder weights: missing={sorted(missing)}, unexpected={sorted(extra)}")
        if set(head) != {"weight", "bias"}:
            raise ValueError("Head weights must contain exactly weight and bias")
        converted = {mapping[k]: v.astype(dtype) for k, v in weights.items()}
        converted.update({f"head.{k}": v.astype(mx.float32) for k, v in head.items()})
        expected = dict(tree_flatten(self.parameters()))
        for key, value in converted.items():
            if value.shape != expected[key].shape:
                raise ValueError(f"{key}: expected shape {expected[key].shape}, got {value.shape}")
        self.load_weights(list(converted.items()), strict=True)
        self.eval()
        mx.eval(self.parameters())

    @classmethod
    def from_folder(cls, folder: Path, dtype=mx.float32):
        cfg = SimpleNamespace(**json.loads((Path(folder) / "config.json").read_text()))
        model = cls(cfg)
        model.load_checkpoint(mx.load(str(Path(folder) / "model.safetensors")),
                              mx.load(str(Path(folder) / "head.safetensors")), dtype)
        return model


class StateCache:
    """Least recently used state tensors, bounded by their storage bytes."""

    def __init__(self, max_bytes=512 * 2**20):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = max_bytes
        self._items = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __contains__(self, key):
        return key in self._items

    def __len__(self):
        return len(self._items)

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return item[0], item[1]

    def put(self, key, ks, vs):
        size = sum(t.nbytes for t in ks + vs)
        if size > self.max_bytes:
            return
        mx.eval(ks, vs)
        if key in self._items:
            self.bytes -= self._items.pop(key)[2]
        self._items[key] = (ks, vs, size)
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, (_, _, evicted) = self._items.popitem(last=False)
            self.bytes -= evicted
            self.evictions += 1

    def clear(self):
        self._items.clear()
        self.bytes = 0
