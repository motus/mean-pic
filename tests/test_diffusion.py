from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from mean_pic.api import ImageEmbedding
from mean_pic.diffusion import (
    DiffusionImageModel,
    DiffusionImageSettings,
    _scale_latents,
    _unscale_latents,
    fit_image,
)


class FakeLatentDistribution:
    std = torch.full((1, 4, 2, 2), 0.4)

    def mode(self) -> torch.Tensor:
        return torch.full((1, 4, 2, 2), 2.0)


class FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.25)

    def __init__(self) -> None:
        self.decoded = None

    def encode(self, pixels):
        return SimpleNamespace(latent_dist=FakeLatentDistribution())

    def decode(self, latents, return_dict=False):
        self.decoded = latents
        return (torch.zeros(1, 3, 16, 16),)


class FakeImageProcessor:
    def preprocess(self, image, *, height, width):
        assert image.size == (width, height)
        return torch.zeros(1, 3, height, width)

    def postprocess(self, values, *, output_type):
        assert output_type == "pil"
        return [Image.new("RGB", (16, 16), "blue")]


class FakePipeline:
    vae_scale_factor = 2

    def __init__(self) -> None:
        self.unet = SimpleNamespace(config=SimpleNamespace(sample_size=8))
        self.vae = FakeVAE()
        self.image_processor = FakeImageProcessor()
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(images=[Image.new("RGB", (16, 16), "red")])


def test_fit_image_resizes_and_crops() -> None:
    result = fit_image(Image.new("RGBA", (40, 20), "white"), 16, 16)

    assert result.mode == "RGB"
    assert result.size == (16, 16)


def test_image_to_embedding_uses_deterministic_vae_mode() -> None:
    model = DiffusionImageModel(
        DiffusionImageSettings(),
        pipeline=FakePipeline(),
    )

    result = model.image_to_embedding(Image.new("RGB", (5, 20)))

    assert result.values.shape == (4, 2, 2)
    assert torch.equal(result.values, torch.full((4, 2, 2), 0.875))
    assert torch.allclose(result.noise, torch.full((4, 2, 2), 0.2))


def test_zero_steps_decodes_mean_latent_without_diffusion() -> None:
    pipeline = FakePipeline()
    model = DiffusionImageModel(
        DiffusionImageSettings(),
        pipeline=pipeline,
    )

    result = model.embedding_to_image(
        ImageEmbedding(
            torch.full((4, 2, 2), 0.875),
            torch.full((4, 2, 2), 0.2),
        ),
        max_iterations=0,
    )

    assert result.getpixel((0, 0)) == (0, 0, 255)
    assert pipeline.call is None
    assert torch.equal(pipeline.vae.decoded, torch.full((1, 4, 2, 2), 2.0))


def test_positive_steps_refine_latent() -> None:
    pipeline = FakePipeline()
    model = DiffusionImageModel(
        DiffusionImageSettings(prompt="quality", strength=0.25),
        pipeline=pipeline,
    )
    embedding = ImageEmbedding(torch.zeros(4, 2, 2), torch.ones(4, 2, 2))

    result = model.embedding_to_image(
        embedding,
        prompt="a room",
        max_iterations=12,
    )

    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert pipeline.call["prompt"] == "a room\n\nquality"
    assert pipeline.call["num_inference_steps"] == 12
    assert pipeline.call["strength"] == 0.25
    assert pipeline.call["image"].shape == (1, 4, 2, 2)


def test_scale_and_unscale_latents_round_trip() -> None:
    config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.25)
    values = torch.tensor([1.0, 2.0])

    assert torch.allclose(
        _unscale_latents(_scale_latents(values, config), config),
        values,
    )


@pytest.mark.parametrize("steps", [-1, True])
def test_model_rejects_invalid_steps(steps) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DiffusionImageModel(
            DiffusionImageSettings(max_denoising_steps=steps),
            pipeline=FakePipeline(),
        )
