from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageEmbedding:
    """An image latent and its component-wise posterior standard deviation."""

    values: torch.Tensor
    noise: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("embedding must have shape [channels, height, width]")
        if self.noise.shape != self.values.shape:
            raise ValueError("noise must have the same shape as embedding values")
        if not self.values.is_floating_point():
            raise TypeError("embedding values must be floating point")
        if not self.noise.is_floating_point():
            raise TypeError("noise values must be floating point")
        if not torch.isfinite(self.values).all() or not torch.isfinite(
            self.noise
        ).all():
            raise ValueError("embedding values and noise must be finite")
        if not torch.all(self.noise > 0):
            raise ValueError("noise values must be positive")


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

    values = torch.stack([item.values.float() for item in embeddings])
    noise = torch.stack([item.noise.float() for item in embeddings])
    precision = noise.square().reciprocal()
    total_precision = precision.sum(dim=0)
    weighted_values = (values * precision).sum(dim=0) / total_precision
    combined_noise = total_precision.rsqrt()
    return ImageEmbedding(weighted_values, combined_noise)


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
