"""Encode both named heads using vLLM's dtype and byte-order conventions."""

from __future__ import annotations

import json
import sys
from copy import copy
from typing import TYPE_CHECKING, TypedDict, cast

from starlette.responses import JSONResponse
from vllm.entrypoints.pooling.utils import (
    build_pooling_bytes_streaming_response,
    encode_pooling_output_base64,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from starlette.responses import Response
    from vllm.entrypoints.pooling.types import PoolingServeContext
    from vllm.outputs import PoolingRequestOutput

    from .io import JointIOProcessor


class EncodedHead(TypedDict):
    """A row-major tensor encoded as base64, with its reconstruction metadata."""

    data: str
    shape: list[int]
    embed_dtype: str
    endianness: str


def encoded_response(processor: JointIOProcessor, ctx: PoolingServeContext) -> Response:
    """Serialize named heads without constructing Python float lists."""
    request = ctx.request
    byteorder = sys.byteorder if request.endianness == "native" else request.endianness
    rows: list[PoolingRequestOutput] = []
    labels: list[tuple[int, str]] = []
    named: list[dict[str, EncodedHead]] = []
    for index, (original, heads) in enumerate(
        zip(ctx.final_res_batch, processor.tensor_outputs(ctx.final_res_batch), strict=True)
    ):
        entry = {}
        for name, tensor in cast("Mapping[str, torch.Tensor]", heads).items():
            row = copy(original)
            row.outputs = copy(original.outputs)
            row.outputs.data = tensor.detach().cpu()
            # Native usage accounting must count the prompt once, not per head.
            row.prompt_token_ids = original.prompt_token_ids if name == "dense" else []
            rows.append(row)
            labels.append((index, name))
            if request.encoding_format == "base64":
                entry[name] = EncodedHead(
                    data=encode_pooling_output_base64(row, request.embed_dtype, request.endianness),
                    shape=list(tensor.shape),
                    embed_dtype=request.embed_dtype,
                    endianness=byteorder,
                )
        named.append(entry)
    if request.encoding_format == "base64":
        return JSONResponse(
            {
                "request_id": ctx.request_id,
                "created_at": ctx.created_time,
                "data": named,
            }
        )
    response = build_pooling_bytes_streaming_response(
        pooling_outputs=rows,
        request_id=ctx.request_id,
        created_time=ctx.created_time,
        model_name=ctx.model_name,
        encoding_format=request.encoding_format,
        embed_dtype=request.embed_dtype,
        endianness=request.endianness,
    )
    if request.encoding_format == "bytes":
        metadata = json.loads(response.headers["metadata"])
        for item, (index, name) in zip(metadata["data"], labels, strict=True):
            item.update(input_index=index, head=name, endianness=byteorder)
        response.headers["metadata"] = json.dumps(metadata)
    return response
