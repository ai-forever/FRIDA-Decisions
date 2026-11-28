"""Read FRIDA binary v1 responses in Python benchmark clients."""

import json
from struct import Struct
from typing import Any

import numpy as np

MEDIA_TYPE = "application/vnd.frida.embeddings.v1"
LENGTH = Struct("<I")


def decode(body: bytes) -> dict[str, Any]:
    """Read the little-endian descriptor length and view the FP32 payload in place."""
    (header_size,) = LENGTH.unpack_from(body)
    start = LENGTH.size
    payload_start = start + header_size
    metadata = json.loads(body[start:payload_start])
    if metadata["dtype"] != "<f4":
        raise ValueError("Invalid FRIDA embedding dtype")
    data = memoryview(body)[payload_start:]
    offset = 0
    decoded = []
    for item in metadata["data"]:
        heads = {}
        for name in ("dense", "multivector"):
            part = item[name]
            lo, hi = part["start"], part["end"]
            if lo != offset or hi < lo or hi > len(data):
                raise ValueError("Invalid FRIDA embedding offsets")
            heads[name] = np.frombuffer(data[lo:hi], dtype="<f4").reshape(part["shape"])
            offset = hi
        decoded.append(heads)
    if offset != len(data):
        raise ValueError("Unexpected trailing bytes in FRIDA response")
    return {**metadata, "data": decoded}
