"""App-scoped joint embeddings on vLLM's standard POST /pooling route."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import (  # noqa: TC002 - FastAPI inspects route annotations.
    APIRouter,
    Depends,
    Request,
    Response,
)
from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor, TokenEmbedIOProcessor
from vllm.entrypoints.pooling.pooling.api_router import pooling
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingRequest,  # noqa: TC002 - FastAPI inspects the request schema.
)
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.exceptions import VLLMValidationError

from .encoding import encoded_response
from .io import JointIOProcessor
from .native import JointInputProcessor, unwrap_data_request
from .transport import joint_response
from .wire import MEDIA_TYPE

if TYPE_CHECKING:
    from argparse import Namespace

    from fastapi import FastAPI
    from starlette.datastructures import State
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
    from vllm.entrypoints.pooling.types import PoolingServeContext


class JointServingPooling(ServingPooling):
    """Use the stock serving lifecycle with processors local to this handler."""

    def __init__(self, stock: ServingPooling) -> None:
        super().__init__(
            stock.engine_client,
            stock.models,
            supported_tasks=stock.supported_tasks,
            request_logger=stock.request_logger,
            chat_template_config=stock.chat_template_config,
        )
        self.joint = JointIOProcessor(self.vllm_config, self.renderer)
        config = (self.vllm_config, self.renderer, self.chat_template_config)
        self.io_processors["plugin"] = JointInputProcessor(*config, io_processor=self.joint)
        for task, processor in (
            ("embed", EmbedIOProcessor),
            ("token_embed", TokenEmbedIOProcessor),
        ):
            if task in self.supported_tasks and task not in self.io_processors:
                self.io_processors[task] = processor(*config)

    def get_io_processor(self, request: PoolingRequest) -> PoolingIOProcessor:
        """Allow both worker tasks while preserving vLLM's default and error paths."""
        if request.task in {"embed", "token_embed"} and request.task in self.supported_tasks:
            if request.dimensions is not None:
                raise VLLMValidationError(
                    "dimensions is currently not supported", parameter="dimensions"
                )
            return self.io_processors[request.task]
        return super().get_io_processor(request)

    def _build_response(self, ctx: PoolingServeContext) -> Response:
        if ctx.request.task != "plugin":
            return super()._build_response(ctx)
        binary = ctx.raw_request is not None and ctx.raw_request.headers.get("accept") == MEDIA_TYPE
        if binary or ctx.request.encoding_format == "float":
            return joint_response(self.joint, ctx, binary=binary)
        response = encoded_response(self.joint, ctx)
        response.headers.add_vary_header("Accept")
        return response


router = APIRouter()


@router.post(
    "/pooling",
    dependencies=[Depends(validate_json_request)],
    responses={
        status: {"model": ErrorResponse}
        for status in (HTTPStatus.BAD_REQUEST, HTTPStatus.INTERNAL_SERVER_ERROR)
    },
)
@with_cancellation
@load_aware_call
async def joint_pooling(request: PoolingRequest, raw_request: Request) -> Response:
    """Dispatch native requests while preserving core behavior on unrelated models."""
    handler = getattr(raw_request.app.state, "joint_serving_pooling", None)
    if handler is None:
        return await pooling(raw_request)(request, raw_request)
    if isinstance(request, IOProcessorRequest):
        request = unwrap_data_request(request, handler.joint.parse_data(request.data))
    return await handler(request, raw_request)


class JointPoolingEndpoint:
    """Install the joint handler for this architecture on the opted-in app only."""

    name = "joint_embeddings"
    required_tasks = ("embed", "token_embed")

    @staticmethod
    def attach_router(app: FastAPI) -> None:
        """Register the replacement POST /pooling route on this app."""
        app.include_router(router)

    @staticmethod
    async def init_state(engine_client: EngineClient | None, state: State, args: Namespace) -> None:
        """Install a handler using the serving configuration of this app."""
        stock = getattr(state, "serving_pooling", None)
        if stock is not None and stock.model_config.architecture == "Qwen35JointEmbedding":
            state.joint_serving_pooling = JointServingPooling(stock)
