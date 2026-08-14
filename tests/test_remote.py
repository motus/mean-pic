from io import BytesIO
import json
from typing import Any

import torch
from PIL import Image

from mean_pic.api import ImageEmbedding
from mean_pic.remote import (
    PROTOCOL_VERSION,
    RemoteLatentImageModel,
    RemoteModelSettings,
)


def test_remote_model_operations() -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    def transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        requests.append((operation, payload))
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "model_id": "example/model",
        }
        if operation == "encode":
            return {
                **metadata,
                "values": [[[1.0]]],
                "noise": [[[0.5]]],
            }
        from mean_pic.image_io import encode_image

        return {
            **metadata,
            "image": encode_image(Image.new("RGB", (2, 2), "green")),
        }

    model = RemoteLatentImageModel(
        RemoteModelSettings("https://example.test", "example/model"),
        transport=transport,
    )
    image = Image.new("RGB", (2, 2), "blue")

    embedding = model.image_to_embedding(image)
    decoded = model.embedding_to_image(
        embedding,
        prompt="blend",
        max_iterations=0,
    )
    interpolated = model.interpolate_images(
        image,
        image,
        prompt="blend",
        max_iterations=4,
    )

    assert torch.equal(embedding.values, torch.tensor([[[1.0]]]))
    assert torch.equal(embedding.noise, torch.tensor([[[0.5]]]))
    assert decoded.size == (2, 2)
    assert interpolated.size == (2, 2)
    assert requests[1][1]["max_iterations"] == 0
    assert requests[2][1]["max_iterations"] == 4


def test_remote_model_sends_bearer_token(monkeypatch) -> None:
    captured = None

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def urlopen(request, timeout):
        nonlocal captured
        captured = request
        return Response(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "model_id": "example/model",
                    "values": [[[1.0]]],
                    "noise": [[[0.5]]],
                }
            ).encode()
        )

    monkeypatch.setattr("mean_pic.remote.urlopen", urlopen)
    model = RemoteLatentImageModel(
        RemoteModelSettings(
            "https://example.test",
            "example/model",
            api_token="secret",
        )
    )

    model.image_to_embedding(Image.new("RGB", (1, 1)))

    assert captured.get_header("Authorization") == "Bearer secret"
