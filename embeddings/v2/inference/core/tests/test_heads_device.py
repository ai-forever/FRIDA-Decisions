from __future__ import annotations

from pathlib import Path

import pytest
import torch
from inference_plugins.contract import Contract
from inference_plugins.heads import JointHeads, load_projection
from safetensors.torch import save_file


def test_explicit_device_overrides_sidecar_cpu(tmp_path: Path) -> None:
    c = Contract(dense_dim=4, multivector_dim=2)
    path = tmp_path / "projection.safetensors"
    save_file({"weight": torch.ones(2, 4)}, path)
    weight = load_projection(path, c)
    assert weight.device.type == "cpu"
    # A distinct device that is available on CPU-only CI too. Real compute is
    # exercised separately on CUDA, not inferred from a meta tensor.
    head = JointHeads(c, weight, device="meta")
    assert head.projection.weight.device.type == "meta"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_loaded_head_matches_backbone_device_and_values(tmp_path: Path, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    c = Contract(dense_dim=4, multivector_dim=2)
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    path = tmp_path / "projection.safetensors"
    save_file({"weight": weight}, path)
    heads = JointHeads(c, load_projection(path, c), device=device)
    hidden = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], device=device)
    dense, multivector = heads(hidden, [1, 2])
    assert dense.device == hidden.device == multivector.device == heads.projection.weight.device
    torch.testing.assert_close(heads.projection.weight.cpu(), weight)
    torch.testing.assert_close(dense, torch.nn.functional.normalize(hidden[-1], dim=-1))
    torch.testing.assert_close(
        multivector, torch.nn.functional.normalize(hidden @ weight.to(device).T, dim=-1)
    )
