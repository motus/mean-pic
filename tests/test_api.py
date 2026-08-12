import pytest
import torch
from PIL import Image

from mean_pic.api import (
    ImageEmbedding,
    interpolate_images,
    mean_embeddings,
)


class FakeModel:
    def image_to_embedding(self, image: Image.Image) -> ImageEmbedding:
        value = float(image.getpixel((0, 0))[0])
        return ImageEmbedding(torch.full((2, 2, 2), value))

    def embedding_to_image(
        self,
        embedding: ImageEmbedding,
        *,
        prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> Image.Image:
        assert prompt == "blend"
        assert max_iterations == 0
        value = round(float(embedding.values.mean()))
        return Image.new("RGB", (2, 2), (value, value, value))


def test_mean_embeddings() -> None:
    left = ImageEmbedding(torch.zeros(2, 2, 2))
    right = ImageEmbedding(torch.full((2, 2, 2), 4.0))

    result = mean_embeddings(left, right)

    assert torch.equal(result.values, torch.full((2, 2, 2), 2.0))


def test_mean_embeddings_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_embeddings(
            ImageEmbedding(torch.zeros(2, 2, 2)),
            ImageEmbedding(torch.zeros(2, 3, 2)),
        )


def test_interpolate_images_uses_two_call_api() -> None:
    result = interpolate_images(
        FakeModel(),
        Image.new("RGB", (1, 1), (10, 0, 0)),
        Image.new("RGB", (1, 1), (30, 0, 0)),
        prompt="blend",
        max_iterations=0,
    )

    assert result.getpixel((0, 0)) == (20, 20, 20)
