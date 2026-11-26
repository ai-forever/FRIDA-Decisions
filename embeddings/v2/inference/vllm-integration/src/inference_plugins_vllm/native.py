"""Render joint HTTP requests through the native pooling input processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.pooling.protocol import PoolingChatRequest, PoolingCompletionRequest

if TYPE_CHECKING:
    from vllm.entrypoints.pooling.pooling.protocol import IOProcessorRequest
    from vllm.entrypoints.pooling.types import AnyPoolingRequest
    from vllm.pooling_params import PoolingParams

    from .io import JointIOProcessor


def unwrap_data_request(
    request: IOProcessorRequest, data: dict[str, Any]
) -> PoolingChatRequest | PoolingCompletionRequest:
    """Use native request validation before normal server context initialization."""
    payload = {**data, **request.model_dump(exclude={"data"}, exclude_unset=True), "task": "plugin"}
    cls = PoolingChatRequest if "messages" in data else PoolingCompletionRequest
    return cls.model_validate(payload)


class JointInputProcessor(PoolingIOProcessor):
    """Render native chat and completion inputs and select both embedding heads."""

    name = "plugin"

    def __init__(self, *args: Any, io_processor: JointIOProcessor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.io_processor = io_processor

    def create_pooling_params(self, request: AnyPoolingRequest) -> PoolingParams:
        """Select token execution with the joint-output flag."""
        return self.io_processor.merge_pooling_params(super().create_pooling_params(request))
