from pathlib import Path

import pytest

from mean_pic.cli import build_model, build_parser
from mean_pic.diffusion import DEFAULT_MODEL_ID, DEFAULT_PROMPT
from mean_pic.remote import RemoteLatentImageModel


def test_cli_defaults() -> None:
    args = build_parser().parse_args(["a.jpg", "b.png", "out.jpg"])

    assert args.first == Path("a.jpg")
    assert args.second == Path("b.png")
    assert args.output == Path("out.jpg")
    assert args.model_id == DEFAULT_MODEL_ID
    assert args.generation_prompt == DEFAULT_PROMPT
    assert args.steps == 48
    assert args.prompt is None


def test_cli_accepts_zero_steps_and_prompt() -> None:
    args = build_parser().parse_args(
        [
            "a.jpg",
            "b.png",
            "out.png",
            "--steps",
            "0",
            "--prompt",
            "a coherent hybrid",
        ]
    )

    assert args.steps == 0
    assert args.prompt == "a coherent hybrid"


def test_cli_builds_remote_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_TOKEN", "secret")
    args = build_parser().parse_args(
        [
            "a.jpg",
            "b.jpg",
            "out.jpg",
            "--backend",
            "remote",
            "--endpoint-url",
            "https://example.test",
            "--api-token-env",
            "TEST_TOKEN",
        ]
    )

    model = build_model(args)

    assert isinstance(model, RemoteLatentImageModel)
    assert model.settings.api_token == "secret"


def test_cli_requires_remote_endpoint() -> None:
    args = build_parser().parse_args(
        ["a.jpg", "b.jpg", "out.jpg", "--backend", "remote"]
    )

    with pytest.raises(ValueError, match="--endpoint-url"):
        build_model(args)
