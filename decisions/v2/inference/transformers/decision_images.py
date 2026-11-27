"""Decode the images field of a FRIDA decision request for standalone inference."""
import base64
import binascii
import io
import json

from PIL import Image


def extract_images(request):
    """Return the decision request and its RGB images in wire order."""
    if isinstance(request, str):
        request = json.loads(request)
    if not isinstance(request, dict) or "images" not in request:
        return request, []
    request = dict(request)
    values = request.pop("images")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("images must be a list of base64 strings")
    if len(values) > 16:
        raise ValueError("at most 16 images per request")

    images = []
    for index, value in enumerate(values):
        if value.startswith("data:"):
            value = value.split(",", 1)[-1]
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"images[{index}] is not valid base64") from error
        try:
            with Image.open(io.BytesIO(raw)) as image:
                images.append(image.convert("RGB"))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"images[{index}] could not be decoded as an image") from error
    return request, images
