"""Released model parity against live CPU FP32 packed inference."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import flat_margins, record

mx = pytest.importorskip("mlx.core")
from frida_decisions import MlxJudge
from frida_decisions.packing import pack
from frida_decisions.protocol import aggregate, decision


@pytest.fixture(scope="module")
def references(torch_judge, cases):
    out = []
    for case in cases:
        p, c, _ = torch_judge.compile(case["request"])
        m = torch_judge.margins_packed(case["request"])
        out.append((m, aggregate(p, c, m)))
    return out


def same_structure(a, b):
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            same_structure(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_structure(x, y)
    elif not isinstance(a, (int, float)):
        assert a == b


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_released_parity(model_dir, cases, references, dtype):
    judge = MlxJudge.from_pretrained(model_dir, dtype=getattr(mx, dtype))
    assert len(judge.model.layers) == 24
    assert judge.model.embed.weight.shape == (93651, 1536)
    assert judge.model.head.weight.shape == (1, 1536)
    assert judge.model.head.weight.dtype == mx.float32
    worst, mismatches, total = 0.0, [], 0
    per_case = {}
    for case, (ref, answers) in zip(cases, references):
        response = judge(case["request"])
        actual = flat_margins(response)
        assert len(actual) == len(ref)
        assert np.isfinite(actual).all()
        assert list(response) == ["model", "answers", "margins", "usage"]
        assert response["usage"]["backend"] == "mlx"
        same_structure(response["answers"], answers)
        _, candidates, _ = judge.compile(case["request"])
        assert [(q, opt) for q, values in response["margins"].items() for opt in values] == [
            (c.question_id, c.option_id) for c in candidates]
        drift = float(np.max(np.abs(np.array(actual) - ref)))
        per_case[case["name"]] = drift
        worst = max(worst, drift)
        for q, answer in answers.items():
            total += 1
            ours = response["answers"][q]
            if decision(answer) != decision(ours):
                mismatches.append(f"{case["name"]}:{q}")
            if answer["type"] == "ranking":
                assert answer["ranking"] == ours["ranking"]
    record(f"mlx_{dtype}_parity", {"requests": len(cases), "decisions": total,
                                  "max_margin_drift": worst, "mismatches": mismatches,
                                  "per_case": per_case})
    assert not mismatches
    if dtype == "float32":
        assert worst <= 1e-3   # Investigate before accepting drift above the target.
    del judge
    mx.clear_cache()


def test_batch_and_row_chunks(model_dir, cases, references):
    judge = MlxJudge.from_pretrained(model_dir, rows_per_forward=2)
    assert judge.judge_batch([]) == []
    responses = judge.judge_batch([c["request"] for c in cases])
    for response, (ref, answers) in zip(responses, references):
        np.testing.assert_allclose(flat_margins(response), ref, atol=1e-3, rtol=0)
        assert {q: decision(a) for q, a in response["answers"].items()} == {
            q: decision(a) for q, a in answers.items()}
    judge.rows_per_forward = None
    actual = judge.margins_packed(cases[-2]["request"])
    np.testing.assert_allclose(actual, references[-2][0], atol=1e-3, rtol=0)


def test_intermediate_encoder_parity(model_dir, torch_judge, cases):
    import torch
    judge = MlxJudge.from_pretrained(model_dir)
    req = cases[0]["request"]
    assert judge.compile(req)[2] == torch_judge.compile(req)[2]
    batch = pack([judge.compile(req)[2]], judge.config)
    ids = torch.from_numpy(batch.input_ids)
    buckets = torch.from_numpy(batch.buckets.astype(np.int64))
    allowed = torch.from_numpy(batch.allowed)
    model, ref = judge.model, torch_judge.model
    x = model.embed(mx.array(batch.input_ids, dtype=mx.int32))
    bias = model.attention_bias(mx.array(batch.buckets), mx.array(batch.allowed))
    report = {}
    def compare(name, x, y):
        actual, expected = np.array(x.astype(mx.float32)), y.float().numpy()
        drift = float(np.max(np.abs(actual - expected)))
        scale = max(float(np.max(np.abs(expected))), 1.0)
        report[name] = {"max_absolute_drift": drift, "relative_to_maximum": drift / scale}
        # Residual states contain values near 1e6. Subtracting attention from
        # them amplifies absolute cancellation error before the final norm.
        if name in ("hidden", "pool", "head"):
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
        else:
            assert drift / scale <= 1e-5, (name, drift, scale)
    with torch.inference_mode():
        y = ref.embed(ids)
        tbias = ref.attention_bias(buckets, allowed)
        compare("embedding", x, y)
        compare("bias", bias, tbias)
        for i, (layer, block) in enumerate(zip(model.layers, ref.blocks)):
            attn = block.layer[0]
            q, k, v = ref._qkv(attn.SelfAttention, attn.layer_norm(y))
            a = ref._attend(attn.SelfAttention, q, k, v, tbias)
            ma = layer.attention(layer.ln1(x), bias)
            if i == 0:
                compare("attention_0", ma, a)
            y = block.layer[-1](y + a)
            x = x + ma
            x = x + layer.dense(layer.ln2(x))
            if i in (0, 5, 11, 23):
                compare(f"block_{i}", x, y)
        y = ref.final_norm(y)
        x = model.final_norm(x)
        compare("hidden", x, y)
        r = batch.readout
        picked = y[r.row, r.col].float()
        pooled = torch.zeros((r.count, picked.shape[-1])).index_add_(0, torch.from_numpy(r.slot), picked)
        counts = torch.bincount(torch.from_numpy(r.slot)).float()[:, None]
        pooled /= counts
        mp = model.pool(x, mx.array(r.row), mx.array(r.col), mx.array(r.slot), r.count)
        compare("pool", mp, pooled)
        compare("head", model.head(mp), ref.head(pooled))
    record("mlx_intermediates", report)
