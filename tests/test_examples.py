from pathlib import Path

import pytest
from PIL import Image

EXAMPLES = Path(__file__).parent / "example"


@pytest.mark.parametrize(
    "name",
    ["cat.jpg", "fish.jpg", "person.jpg", "landscape.jpg", "interior.jpg"],
)
def test_example_image_is_a_valid_jpeg(name: str) -> None:
    with Image.open(EXAMPLES / name) as image:
        image.verify()
        assert image.format == "JPEG"
