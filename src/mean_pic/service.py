from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from mean_pic.api import ImageEmbedding, LatentImageModel, interpolate_images
from mean_pic.image_io import decode_image, encode_image
from mean_pic.remote import PROTOCOL_VERSION


class LatentImageService:
    def __init__(self, model: LatentImageModel, model_id: str) -> None:
        self.model = model
        self.model_id = model_id

    def handle(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_metadata(payload)
        if operation == "encode":
            result = {
                "values": self.model.image_to_embedding(
                    decode_image(payload.get("image"))
                ).values.float().tolist()
            }
        elif operation == "decode":
            try:
                embedding = ImageEmbedding(
                    torch.tensor(payload["values"], dtype=torch.float32)
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("values must be a finite image latent") from error
            result = {
                "image": encode_image(
                    self.model.embedding_to_image(
                        embedding,
                        prompt=self._prompt(payload),
                        max_iterations=self._max_iterations(payload),
                    )
                )
            }
        elif operation == "interpolate":
            encoded_images = payload.get("images")
            if not isinstance(encoded_images, list) or not encoded_images:
                raise ValueError("images must be a non-empty list")
            result = {
                "image": encode_image(
                    interpolate_images(
                        self.model,
                        *(decode_image(value) for value in encoded_images),
                        prompt=self._prompt(payload),
                        max_iterations=self._max_iterations(payload),
                    )
                )
            }
        else:
            raise ValueError(f"unsupported operation: {operation}")

        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": self.model_id,
            **result,
        }

    def _validate_metadata(self, payload: dict[str, Any]) -> None:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol version does not match")
        if payload.get("model_id") != self.model_id:
            raise ValueError("model ID does not match")

    @staticmethod
    def _prompt(payload: dict[str, Any]) -> str | None:
        prompt = payload.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        return prompt

    @staticmethod
    def _max_iterations(payload: dict[str, Any]) -> int | None:
        value = payload.get("max_iterations")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError("max_iterations must be a non-negative integer")
        return value


def create_app(service: LatentImageService):
    fastapi = import_module("fastapi")
    app = fastapi.FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": service.model_id,
        }

    @app.post("/{operation}")
    def execute(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.handle(operation, payload)
        except ValueError as error:
            raise fastapi.HTTPException(status_code=400, detail=str(error)) from error

    return app
