import base64
from io import BytesIO

import pytest
from PIL import Image

from mean_pic.image_io import decode_image, encode_image


def test_image_base64_round_trip() -> None:
    encoded = encode_image(Image.new("RGB", (3, 2), (1, 2, 3)))

    result = decode_image(encoded)

    assert result.size == (3, 2)
    assert result.getpixel((0, 0)) == (1, 2, 3)


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_decode_image_accepts_supported_formats(format: str) -> None:
    buffer = BytesIO()
    Image.new("RGB", (5, 3), "purple").save(buffer, format=format)

    result = decode_image(base64.b64encode(buffer.getvalue()).decode("ascii"))

    assert result.mode == "RGB"
    assert result.size == (5, 3)
