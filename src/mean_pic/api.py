from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageEmbedding:
    """A single image latent with shape [channels, height, width]."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("embedding must have shape [channels, height, width]")
        if not self.values.is_floating_point():
            raise TypeError("embedding values must be floating point")
        if not torch.isfinite(self.values).all():
            raise ValueError("embedding values must be finite")


class LatentImageModel(Protocol):
    def image_to_embedding(self, image: Image.Image) -> ImageEmbedding: ...

    def embedding_to_image(
        self,
        embedding: ImageEmbedding,
        *,
        prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> Image.Image: ...


def mean_embeddings(*embeddings: ImageEmbedding) -> ImageEmbedding:
    if not embeddings:
        raise ValueError("at least one embedding is required")

    shape = embeddings[0].values.shape
    if any(item.values.shape != shape for item in embeddings[1:]):
        raise ValueError("all embeddings must have the same shape")

    values = torch.stack([item.values.float() for item in embeddings]).mean(dim=0)
    return ImageEmbedding(values)


def interpolate_images(
    model: LatentImageModel,
    *images: Image.Image,
    prompt: str | None = None,
    max_iterations: int | None = None,
) -> Image.Image:
    embedding = mean_embeddings(
        *(model.image_to_embedding(image) for image in images)
    )
    return model.embedding_to_image(
        embedding,
        prompt=prompt,
        max_iterations=max_iterations,
    )
