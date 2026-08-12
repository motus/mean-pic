from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen

import torch
from PIL import Image

from mean_pic.api import ImageEmbedding
from mean_pic.image_io import decode_image, encode_image

PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class RemoteModelSettings:
    endpoint_url: str
    model_id: str
    api_token: str | None = None
    timeout: float = 300.0


class RemoteLatentImageModel:
    def __init__(
        self,
        settings: RemoteModelSettings,
        *,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport or self._post

    def image_to_embedding(self, image: Image.Image) -> ImageEmbedding:
        response = self._request("encode", {"image": encode_image(image)})
        try:
            values = torch.tensor(response["values"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("remote encode response has invalid embedding") from error
        return ImageEmbedding(values)

    def embedding_to_image(
        self,
        embedding: ImageEmbedding,
        *,
        prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> Image.Image:
        payload: dict[str, Any] = {
            "values": embedding.values.detach().float().cpu().tolist()
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if max_iterations is not None:
            payload["max_iterations"] = max_iterations
        return decode_image(self._request("decode", payload)["image"])

    def interpolate_images(
        self,
        *images: Image.Image,
        prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> Image.Image:
        payload: dict[str, Any] = {
            "images": [encode_image(image) for image in images]
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if max_iterations is not None:
            payload["max_iterations"] = max_iterations
        return decode_image(self._request("interpolate", payload)["image"])

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._transport(
            operation,
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_id": self.settings.model_id,
                **payload,
            },
        )
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("remote protocol version does not match")
        if response.get("model_id") != self.settings.model_id:
            raise ValueError("remote model ID does not match")
        return response

    def _post(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_token:
            headers["Authorization"] = f"Bearer {self.settings.api_token}"
        request = Request(
            f"{self.settings.endpoint_url.rstrip('/')}/{operation}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("remote response must be a JSON object")
        return result
