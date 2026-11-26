"""Fast HTTP responses for both embedding heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson
from starlette.responses import Response, StreamingResponse

from .wire import LENGTH, MEDIA_TYPE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from vllm.entrypoints.pooling.types import PoolingServeContext

    from .io import JointIOProcessor


def joint_response(
    processor: JointIOProcessor, ctx: PoolingServeContext, *, binary: bool
) -> Response:
    """Serialize CPU tensor views, placing FP32 frame descriptors in the body."""
    metadata = {"request_id": ctx.request_id, "created_at": ctx.created_time}
    pairs = (
        {
            "dense": heads["dense"].detach().cpu().float().contiguous().numpy(),
            "multivector": heads["multivector"].detach().cpu().float().contiguous().numpy(),
        }
        for heads in processor.tensor_outputs(ctx.final_res_batch)
    )
    if not binary:
        body = orjson.dumps({**metadata, "data": list(pairs)}, option=orjson.OPT_SERIALIZE_NUMPY)
        return Response(body, media_type="application/json", headers={"Vary": "Accept"})

    offset = 0
    chunks = []
    items = []
    for pair in pairs:
        item = {}
        # The published FRIDA v1 decoder expects "colbert" on the binary wire.
        for field, key in (("dense", "dense"), ("multivector", "colbert")):
            array = pair[field].astype("<f4", copy=False)
            chunk = array.tobytes()
            item[key] = {"start": offset, "end": offset + len(chunk), "shape": array.shape}
            offset += len(chunk)
            chunks.append(chunk)
        items.append(item)
    header = orjson.dumps({**metadata, "dtype": "<f4", "data": items})
    prefix = LENGTH.pack(len(header)) + header

    async def stream() -> AsyncIterator[bytes]:
        yield prefix
        for chunk in chunks:
            yield chunk

    return StreamingResponse(
        stream(),
        media_type=MEDIA_TYPE,
        headers={"Content-Length": str(len(prefix) + offset), "Vary": "Accept"},
    )
