from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import sglang_outputs

CONTRACT = SimpleNamespace(dense_dim=2, multivector_dim=2, normalize_dense=True)


def test_json_response_normalizes_raw_dense() -> None:
    raw = {"embedding": [[1.0, 0.0]], "pooled_hidden_state": [3.0, 4.0]}
    result = sglang_outputs(json.loads(json.dumps(raw)), CONTRACT)
    assert result == [{"dense": [0.6, 0.8], "multivector": [[1.0, 0.0]]}]
    assert raw["pooled_hidden_state"] == [3.0, 4.0]


@pytest.mark.parametrize("dense", [{}, [[3.0, 4.0]], [3.0], "tensor"])
def test_invalid_dense_reports_transport_shape(dense: object) -> None:
    with pytest.raises(ValueError, match="pooled_hidden_state=.*embedding="):
        sglang_outputs({"embedding": [[1.0, 0.0]], "pooled_hidden_state": dense}, CONTRACT)


def test_invalid_multivector_reports_transport_shape() -> None:
    with pytest.raises(ValueError, match="embedding=list\\[2\\] of float"):
        sglang_outputs({"embedding": [1.0, 0.0], "pooled_hidden_state": [3.0, 4.0]}, CONTRACT)
