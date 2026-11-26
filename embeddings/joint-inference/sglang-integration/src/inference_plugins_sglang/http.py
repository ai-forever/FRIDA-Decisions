"""Serialize native embedding tensors before FastAPI's generic object encoder."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI


def tensor_lists(value: object) -> object:
    """Convert response tensors without changing offline engine outputs."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_lists(item) for item in value]
    return value


def install_encode_serialization(app: FastAPI) -> None:
    """Wrap only native /encode; leave other HTTP endpoints and engine IPC alone.

    Updating both endpoint and dependant.call preserves the existing FastAPI
    input validation. Rebuild the ASGI handler after changing the dependency.
    """
    from fastapi.routing import APIRoute, request_response

    found = False
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path != "/encode":
            continue
        found = True
        original = route.dependant.call
        if original is None:
            raise RuntimeError("SGLang /encode endpoint has no callable")
        if getattr(original, "_joint_tensor_serialization", False):
            continue

        def wrap(endpoint: Callable[..., Awaitable[object]]) -> Callable[..., Awaitable[object]]:
            @wraps(endpoint)
            async def encode(*args: Any, **kwargs: Any) -> object:
                return tensor_lists(await endpoint(*args, **kwargs))

            setattr(encode, "_joint_tensor_serialization", True)  # noqa: B010 - function metadata
            return encode

        route.endpoint = route.dependant.call = wrap(original)
        route.app = request_response(route.get_route_handler())
    if not found:
        raise RuntimeError("SGLang /encode route not found; incompatible HTTP server API")
