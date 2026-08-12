from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def encode_image(image: Image.Image, *, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str):
        raise ValueError("image must be a base64 string")
    try:
        data = base64.b64decode(value, validate=True)
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as error:
        raise ValueError("image must be valid base64-encoded image data") from error
    return image.convert("RGB")
