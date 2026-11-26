"""Check real JSON and FP32 response payloads with a full indexing batch."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import torch
from inference_plugins_vllm.transport import joint_response
from inference_plugins_vllm.wire import MEDIA_TYPE, decode


def test_batch_64_transport(monkeypatch):
    heads = [
        {"dense": torch.randn(8, dtype=torch.float16)[::2],
         "multivector": torch.randn(i % 8, 4, dtype=torch.bfloat16)[:, ::2]}
        for i in range(64)
    ]
    processor = SimpleNamespace(tensor_outputs=iter)
    ctx = SimpleNamespace(final_res_batch=heads, request_id="test", created_time=123)
    def no_python_floats(*args):
        raise AssertionError("materialized Python floats")
    monkeypatch.setattr(torch.Tensor, "tolist", no_python_floats)
    response = joint_response(processor, ctx, binary=False)
    values = json.loads(response.body)["data"]
    response = joint_response(processor, ctx, binary=True)
    async def read(): return b"".join([chunk async for chunk in response.body_iterator])
    body = asyncio.run(read())
    assert len(body) == int(response.headers["content-length"])
    assert response.media_type == MEDIA_TYPE and "metadata" not in response.headers
    assert sum(len(k) + len(v) for k, v in response.raw_headers) < 1024
    for row, item, json_row in zip(heads, decode(body)["data"], values, strict=True):
        for key, wire in (("dense", "dense"), ("multivector", "colbert")):
            expected = row[key].float().numpy()
            np.testing.assert_array_equal(expected, item[wire])
            np.testing.assert_allclose(np.asarray(json_row[key]).reshape(expected.shape), expected, rtol=1e-7)
