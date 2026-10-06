"""Small deterministic tests for native MLX computation and strict loading."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from frida_decisions.mlx_modeling import FridaMlxDecisionModel, T5Norm, gelu_new
from frida_decisions.config import DecisionsConfig
from frida_decisions.packing import TokenizedRequest, pack


@pytest.fixture
def tiny():
    cfg = SimpleNamespace(model_type="t5", feed_forward_proj="gated-gelu",
                          dense_act_fn="gelu_new", vocab_size=32, d_model=16,
                          d_ff=32, num_heads=2, d_kv=8, num_layers=2,
                          layer_norm_epsilon=1e-6, relative_attention_num_buckets=32)
    mx.random.seed(5)
    return FridaMlxDecisionModel(cfg)


def checkpoint(model):
    from mlx.utils import tree_flatten
    params = dict(tree_flatten(model.parameters()))
    return ({src: params[dst] for src, dst in model.weight_mapping().items()},
            {k: params[f"head.{k}"] for k in ("weight", "bias")})


def test_loading_and_head_precision(tiny):
    weights, head = checkpoint(tiny)
    tiny.load_checkpoint(weights, head, mx.bfloat16)
    assert tiny.embed.weight.dtype == mx.bfloat16
    assert tiny.layers[0].dense.wo.weight.dtype == mx.bfloat16
    assert tiny.head.weight.dtype == tiny.head.bias.dtype == mx.float32
    assert len(tiny.layers) == 2


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "head", "alias"])
def test_loading_rejects_broken_weights(tiny, fault):
    weights, head = checkpoint(tiny)
    if fault == "missing":
        del weights["encoder.block.1.layer.0.SelfAttention.q.weight"]
    elif fault == "extra":
        weights["decoder.fake.weight"] = mx.zeros((1,))
    elif fault == "shape":
        weights["shared.weight"] = mx.zeros((1, 1))
    elif fault == "head":
        head["weight"] = mx.zeros((2, 16))
    else:
        weights["encoder.embed_tokens.weight"] = weights["shared.weight"] + 1
    with pytest.raises(ValueError):
        tiny.load_checkpoint(weights, head, mx.float32)


@pytest.mark.parametrize("shared", [True, False])
def test_tied_embedding_alias(tiny, shared):
    weights, head = checkpoint(tiny)
    weights["encoder.embed_tokens.weight"] = weights["shared.weight"]
    if not shared:
        del weights["shared.weight"]
    tiny.load_checkpoint(weights, head, mx.float32)


def test_gelu_matches_hugging_face():
    torch = pytest.importorskip("torch")
    from transformers.activations import NewGELUActivation
    x = np.linspace(-8, 8, 1001, dtype=np.float32)
    ref = NewGELUActivation()(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(np.array(gelu_new(mx.array(x))), ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_norm_matches_t5(dtype):
    torch = pytest.importorskip("torch")
    from transformers.models.t5.modeling_t5 import T5LayerNorm
    x = np.random.default_rng(7).normal(size=(2, 5, 16)).astype(np.float32)
    tdtype = torch.float32 if dtype == mx.float32 else torch.bfloat16
    ref = T5LayerNorm(16).to(tdtype)(torch.from_numpy(x).to(tdtype)).float().detach().numpy()
    norm = T5Norm(16, 1e-6)
    norm.weight = norm.weight.astype(dtype)
    actual = np.array(norm(mx.array(x).astype(dtype)).astype(mx.float32))
    np.testing.assert_allclose(actual, ref, atol=1e-6, rtol=1e-6)


def test_candidates_are_isolated_and_positions_restart(tiny):
    req = TokenizedRequest([3, 4], [[5, 6]], [(0, [7, 2]), (0, [8, 9, 2])])
    cfg = DecisionsConfig(align=8)
    b = pack([req], cfg)
    def margins(request):
        b = pack([request], cfg)
        h = tiny(mx.array(b.input_ids, dtype=mx.int32), mx.array(b.buckets), mx.array(b.allowed))
        r = b.readout
        return np.array(tiny.margins(h, mx.array(r.row), mx.array(r.col), mx.array(r.slot), r.count))
    together = margins(req)
    for i, option in enumerate(req.options):
        alone = margins(TokenizedRequest(req.state, req.questions, [option]))
        np.testing.assert_allclose(together[i], alone[0], atol=2e-6, rtol=2e-6)
    changed = TokenizedRequest(req.state, req.questions, [(0, [7, 2]), (0, [20, 21, 22, 2])])
    np.testing.assert_allclose(together[0], margins(changed)[0], atol=2e-6, rtol=2e-6)
    bias = np.array(tiny.attention_bias(mx.array(b.buckets), mx.array(b.allowed)))
    assert np.all(bias[0, :, 4:6, 6:9] == np.finfo(np.float32).min)


def test_pool_converts_to_fp32_before_mean(tiny):
    h = mx.array([[[1, 2], [3, 4], [5, 6]]], dtype=mx.bfloat16)
    p = tiny.pool(h, mx.array([0, 0, 0]), mx.array([0, 1, 2]), mx.array([0, 0, 1]), 2)
    assert p.dtype == mx.float32
    np.testing.assert_array_equal(np.array(p), [[2, 3], [5, 6]])
