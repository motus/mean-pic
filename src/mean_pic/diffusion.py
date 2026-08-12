from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageOps

from mean_pic.api import ImageEmbedding

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPT = ""


@dataclass(frozen=True, slots=True)
class DiffusionImageSettings:
    model_id: str = DEFAULT_MODEL_ID
    prompt: str = DEFAULT_PROMPT
    max_denoising_steps: int = 48
    strength: float = 0.5
    guidance_scale: float = 7.5
    device: str = "auto"
    dtype: str = "auto"
    cpu_offload: bool = False


class DiffusionImageModel:
    """Stable Diffusion image-to-latent and latent-to-image adapter."""

    def __init__(
        self,
        settings: DiffusionImageSettings | None = None,
        *,
        pipeline: Any | None = None,
    ) -> None:
        self.settings = settings or DiffusionImageSettings()
        _validate_steps(self.settings.max_denoising_steps)
        if not 0 < self.settings.strength <= 1:
            raise ValueError("strength must be in (0, 1]")

        self.pipeline = pipeline or self._load_pipeline()
        sample_size = self.pipeline.unet.config.sample_size
        if isinstance(sample_size, int):
            sample_height = sample_width = sample_size
        else:
            sample_height, sample_width = sample_size
        self.height = int(sample_height) * int(self.pipeline.vae_scale_factor)
        self.width = int(sample_width) * int(self.pipeline.vae_scale_factor)

    def _load_pipeline(self):
        from diffusers import AutoPipelineForImage2Image

        device = _resolve_device(self.settings.device)
        dtype = _resolve_dtype(self.settings.dtype, device)
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.settings.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        if self.settings.cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
        return pipeline

    @torch.inference_mode()
    def image_to_embedding(self, image: Image.Image) -> ImageEmbedding:
        fitted = fit_image(image, self.width, self.height)
        pixels = self.pipeline.image_processor.preprocess(
            fitted,
            height=self.height,
            width=self.width,
        )
        pixels = pixels.to(
            device=self.pipeline.vae.device,
            dtype=self.pipeline.vae.dtype,
        )
        encoded = self.pipeline.vae.encode(pixels).latent_dist.mode()
        values = _scale_latents(encoded, self.pipeline.vae.config)
        return ImageEmbedding(values[0].detach().float().cpu())

    @torch.inference_mode()
    def embedding_to_image(
        self,
        embedding: ImageEmbedding,
        *,
        prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> Image.Image:
        steps = (
            self.settings.max_denoising_steps
            if max_iterations is None
            else max_iterations
        )
        _validate_steps(steps)
        latents = embedding.values.unsqueeze(0).to(
            device=self.pipeline.vae.device,
            dtype=self.pipeline.vae.dtype,
        )
        if steps == 0:
            decoded = self.pipeline.vae.decode(
                _unscale_latents(latents, self.pipeline.vae.config),
                return_dict=False,
            )[0]
            return self.pipeline.image_processor.postprocess(
                decoded,
                output_type="pil",
            )[0]

        result = self.pipeline(
            prompt=_join_prompt(prompt, self.settings.prompt),
            image=latents,
            num_inference_steps=steps,
            strength=self.settings.strength,
            guidance_scale=self.settings.guidance_scale,
        )
        return result.images[0]


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
    )


def _scale_latents(latents: torch.Tensor, config: Any) -> torch.Tensor:
    shift = getattr(config, "shift_factor", None) or 0.0
    scale = float(config.scaling_factor)
    return (latents - shift) * scale


def _unscale_latents(latents: torch.Tensor, config: Any) -> torch.Tensor:
    shift = getattr(config, "shift_factor", None) or 0.0
    scale = float(config.scaling_factor)
    return latents / scale + shift


def _join_prompt(prompt: str | None, configured_prompt: str) -> str:
    return "\n\n".join(
        part for part in (prompt, configured_prompt) if part
    )


def _validate_steps(steps: int) -> None:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    try:
        return getattr(torch, dtype)
    except AttributeError as error:
        raise ValueError(f"unsupported dtype: {dtype}") from error
