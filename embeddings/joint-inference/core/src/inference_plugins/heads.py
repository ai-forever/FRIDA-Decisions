from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .contract import Contract
    from .types import JointEmbedding, PathLike, TensorEmbedding


MATRIX_NDIM = 2


def load_projection(path: PathLike, contract: Contract) -> torch.Tensor:
    """Load the required bias-free projection and validate its dimensions."""
    weights = load_file(str(path))
    if set(weights) != {"weight"}:
        raise ValueError("Projection must contain exactly 'weight' (no bias)")
    weight = weights["weight"]
    if tuple(weight.shape) != (contract.multivector_dim, contract.dense_dim):
        raise ValueError(f"Projection shape mismatch: {tuple(weight.shape)}")
    return weight


class JointHeads(nn.Module):
    """Compute dense and filtered token embeddings from full-prompt hidden states."""

    def __init__(
        self, contract: Contract, weight: torch.Tensor, *, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.contract = contract
        if tuple(weight.shape) != (contract.multivector_dim, contract.dense_dim):
            raise ValueError("Projection shape mismatch")
        # Safetensors sidecars load on CPU, independently of the engine's
        # default-device context. Place the trained head with its backbone.
        if device is not None:
            weight = weight.to(device=device)
        self.projection = nn.Linear(
            contract.dense_dim,
            contract.multivector_dim,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            self.projection.weight.copy_(weight)

    def dense(self, hidden: torch.Tensor, *, normalize: bool | None = None) -> torch.Tensor:
        """Pool original hidden states without executing the token projection."""
        c = self.contract
        if hidden.ndim != MATRIX_NDIM or hidden.shape[1] != c.dense_dim or not len(hidden):
            raise ValueError("Expected nonempty full-prompt hidden states")
        result = hidden[-1]
        normalize = c.normalize_dense if normalize is None else normalize
        if normalize:
            result = functional.normalize(result, dim=-1)
        return result

    def multivector(
        self,
        hidden: torch.Tensor,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Project token states, optionally normalize them, and filter skipped token IDs."""
        c = self.contract
        if (
            hidden.ndim != MATRIX_NDIM
            or hidden.shape != (len(token_ids), c.dense_dim)
            or not len(token_ids)
        ):
            raise ValueError("Expected nonempty full-prompt hidden states matching token IDs")
        result = self.projection(hidden.to(self.projection.weight.dtype))
        normalize = c.normalize_multivector if normalize is None else normalize
        if normalize:
            result = functional.normalize(result, dim=-1)
        mask = torch.ones(len(token_ids), device=hidden.device, dtype=torch.bool)
        ids = torch.as_tensor(token_ids, device=hidden.device)
        for skip in c.skip_token_ids:
            mask &= ids != skip
        return result[mask]

    def forward(
        self, hidden: torch.Tensor, token_ids: Sequence[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense and filtered token embeddings for one prompt."""
        return self.dense(hidden), self.multivector(hidden, token_ids)


def pack(dense: torch.Tensor, multivector: torch.Tensor, c: Contract) -> torch.Tensor:
    """Combine dense and token embeddings using the joint output layout."""
    if (
        dense.shape != (c.dense_dim,)
        or multivector.ndim != MATRIX_NDIM
        or multivector.shape[1] != c.multivector_dim
    ):
        raise ValueError("Joint output dimensions do not match layout v1")
    return torch.cat(
        (dense.to(multivector.dtype).reshape(-1, c.multivector_dim), multivector), dim=0
    )


def unpack_tensors(packed: torch.Tensor, c: Contract) -> TensorEmbedding:
    """Decode joint layout without converting tensors to Python lists."""
    rows = c.dense_dim // c.multivector_dim
    if packed.ndim != MATRIX_NDIM or packed.shape[1] != c.multivector_dim or packed.shape[0] < rows:
        raise ValueError("Invalid joint output layout v1")
    return {"dense": packed[:rows].reshape(-1), "multivector": packed[rows:]}


def unpack(packed: torch.Tensor, c: Contract) -> JointEmbedding:
    """Decode the joint output layout into named embedding lists."""
    heads = unpack_tensors(packed, c)
    return {"dense": heads["dense"].tolist(), "multivector": heads["multivector"].tolist()}
