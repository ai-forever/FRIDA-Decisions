from __future__ import annotations

import json

import pytest
import torch
from inference_plugins_sglang.http import install_encode_serialization, tensor_lists


def test_tensor_lists_preserves_raw_values_and_source() -> None:
    dense = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    raw = [{"pooled_hidden_state": dense, "embedding": tokens, "meta_info": {"count": 2}}]
    result = tensor_lists(raw)
    assert json.loads(json.dumps(result)) == [
        {
            "pooled_hidden_state": [3.0, 4.0],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "meta_info": {"count": 2},
        }
    ]
    assert raw[0]["pooled_hidden_state"] is dense
    assert raw[0]["embedding"] is tokens


def test_encode_http_roundtrip() -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()

    @app.post("/encode", response_model=None)
    async def encode(payload: dict) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "embedding": torch.tensor([[1.0, 0.0]]),
                "pooled_hidden_state": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            }
            for _ in payload["input_ids"]
        ]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    install_encode_serialization(app)
    install_encode_serialization(app)  # safe to invoke twice
    client = TestClient(app)
    response = client.post("/encode", json={"input_ids": [[1], [2, 3]]})
    assert response.status_code == 200
    assert response.json() == [{"embedding": [[1.0, 0.0]], "pooled_hidden_state": [3.0, 4.0]}] * 2
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/encode", json=[]).status_code == 422
