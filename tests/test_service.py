import pytest
import torch
from PIL import Image

from mean_pic.api import ImageEmbedding
from mean_pic.image_io import decode_image, encode_image
from mean_pic.remote import PROTOCOL_VERSION
from mean_pic.service import LatentImageService


class FakeModel:
    def image_to_embedding(self, image: Image.Image) -> ImageEmbedding:
        return ImageEmbedding(torch.full((1, 1, 1), float(image.width)))

    def embedding_to_image(
        self,
        embedding: ImageEmbedding,
        *,
        prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> Image.Image:
        assert prompt == "blend"
        assert max_iterations == 0
        return Image.new("RGB", (3, 2), "red")


def request(**payload):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "example/model",
        **payload,
    }


def test_service_executes_operations() -> None:
    service = LatentImageService(FakeModel(), "example/model")
    image = encode_image(Image.new("RGB", (4, 2), "blue"))

    encoded = service.handle("encode", request(image=image))
    decoded = service.handle(
        "decode",
        request(
            values=[[[4.0]]],
            prompt="blend",
            max_iterations=0,
        ),
    )
    interpolated = service.handle(
        "interpolate",
        request(
            images=[image, image],
            prompt="blend",
            max_iterations=0,
        ),
    )

    assert encoded["values"] == [[[4.0]]]
    assert decode_image(decoded["image"]).size == (3, 2)
    assert decode_image(interpolated["image"]).size == (3, 2)


@pytest.mark.parametrize(
    ("operation", "payload", "message"),
    [
        ("encode", request(image="invalid"), "base64"),
        ("decode", request(values=[1.0]), "image latent"),
        ("interpolate", request(images=[]), "non-empty"),
        ("decode", request(values=[[[1.0]]], prompt=1), "prompt"),
        ("decode", request(values=[[[1.0]]], max_iterations=-1), "non-negative"),
        ("unknown", request(), "unsupported"),
    ],
)
def test_service_rejects_invalid_requests(operation, payload, message) -> None:
    service = LatentImageService(FakeModel(), "example/model")

    with pytest.raises(ValueError, match=message):
        service.handle(operation, payload)
