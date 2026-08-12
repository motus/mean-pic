from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image

from mean_pic.api import LatentImageModel, interpolate_images
from mean_pic.diffusion import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    DiffusionImageModel,
    DiffusionImageSettings,
)
from mean_pic.remote import RemoteLatentImageModel, RemoteModelSettings


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode the mean latent embedding of two images."
    )
    parser.add_argument("first", type=Path, help="first JPEG or PNG image")
    parser.add_argument("second", type=Path, help="second JPEG or PNG image")
    parser.add_argument("output", type=Path, help="output JPEG or PNG image")
    parser.add_argument(
        "--backend",
        choices=("local", "remote"),
        default="local",
    )
    parser.add_argument("--endpoint-url")
    parser.add_argument(
        "--api-token-env",
        default="MEAN_PIC_API_TOKEN",
    )
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--steps",
        type=non_negative_int,
        default=48,
        help="maximum denoising steps; 0 performs only VAE decoding",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help="UTF-8 text prompt file for diffusion refinement",
    )
    parser.add_argument(
        "--generation-prompt",
        default=DEFAULT_PROMPT,
        help="additional local diffusion prompt",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="offload model components to CPU RAM",
    )
    return parser


def build_model(args: argparse.Namespace) -> LatentImageModel:
    if args.backend == "remote":
        if not args.endpoint_url:
            raise ValueError("--endpoint-url is required with --backend remote")
        return RemoteLatentImageModel(
            RemoteModelSettings(
                endpoint_url=args.endpoint_url,
                model_id=args.model_id,
                api_token=os.environ.get(args.api_token_env),
            )
        )
    return DiffusionImageModel(
        DiffusionImageSettings(
            model_id=args.model_id,
            prompt=args.generation_prompt,
            max_denoising_steps=args.steps,
            cpu_offload=args.cpu_offload,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        model = build_model(args)
        with Image.open(args.first) as first:
            image_a = first.convert("RGB")
        with Image.open(args.second) as second:
            image_b = second.convert("RGB")
        prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else None
        if isinstance(model, RemoteLatentImageModel):
            result = model.interpolate_images(
                image_a,
                image_b,
                prompt=prompt,
                max_iterations=args.steps,
            )
        else:
            result = interpolate_images(
                model,
                image_a,
                image_b,
                prompt=prompt,
                max_iterations=args.steps,
            )
        result.save(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
