# mean-pic

`mean-pic` explores image diffusion latent space in the same style as
[`mean-idea`](https://github.com/motus/mean-idea). It encodes two images with a
model VAE, averages their latent tensors, and converts the mean latent back to
an image.

```console
uv run mean-pic tests/example/cat.jpg tests/example/fish.jpg result.jpg
```

Inputs may be JPEG or PNG and do not need matching dimensions. Each image is
converted to RGB, resized while preserving its aspect ratio, and center-cropped
to the model's native canvas. Pillow selects the output format from the output
filename.

## Installation

Python 3.11-3.13 and a recent PyTorch installation are required.

```console
uv sync
```

The first local run downloads the selected model from Hugging Face.

## CLI

```console
uv run mean-pic FIRST_IMAGE SECOND_IMAGE OUTPUT_IMAGE [options]
```

Important options:

| Option | Meaning |
| --- | --- |
| `--model-id ID` | Diffusers image-to-image model; defaults to Stable Diffusion 1.5 |
| `--steps N` | Maximum diffusion denoising steps; defaults to 48 |
| `--steps 0` | Skip diffusion and directly VAE-decode the averaged latent |
| `--prompt FILE` | Read the diffusion refinement prompt from a UTF-8 file |
| `--generation-prompt TEXT` | Add a local prompt after the file prompt |
| `--cpu-offload` | Save GPU memory by offloading inactive components to CPU RAM |
| `--backend remote --endpoint-url URL` | Perform interpolation through the HTTP API |

The prompt is used only by the latent-to-image diffusion refinement. It is not
embedded into either source image. With `--steps 0`, prompts have no effect.

## Models

The adapter targets Diffusers pipelines supported by
`AutoPipelineForImage2Image` that accept VAE latents as image input. These are
practical choices for a machine with 32 GB GPU RAM and 96 GB CPU RAM:

| Model ID | Native size | Suggested use |
| --- | ---: | --- |
| [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | 512x512 | Default and simplest option; small enough for GPU or CPU RAM |
| [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 1024x1024 | Recommended quality option for the 32 GB GPU |
| [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | 512x512 | Fast GPU experiments; designed for roughly 1-4 steps |

For CPU execution, use Stable Diffusion 1.5; the CLI automatically selects CPU
when CUDA is unavailable. On the server, set `MEAN_PIC_DEVICE=cpu`. The model
fits comfortably in 96 GB RAM but generation is much slower than on a GPU.

FLUX.1 models were considered because quantized/offloaded variants can fit the
available RAM, but they are not selected here: their pipeline and latent
geometry differ from the deliberately small Stable Diffusion adapter.

Latent averaging is experimental. A VAE-decoded mean (`--steps 0`) may look
blurred, while diffusion refinement can replace details from the source images
rather than preserving a literal visual average.

## Python API

```python
from PIL import Image

from mean_pic import interpolate_images
from mean_pic.diffusion import DiffusionImageModel

model = DiffusionImageModel()
with Image.open("cat.jpg") as first, Image.open("fish.png") as second:
    result = interpolate_images(
        model,
        first,
        second,
        prompt="a coherent hybrid subject",
        max_iterations=24,
    )
result.save("result.png")
```

The lower-level API exposes `image_to_embedding()`,
`embedding_to_image()`, and `mean_embeddings()`. `ImageEmbedding.values` is a
floating-point tensor shaped `[latent_channels, latent_height, latent_width]`.

## HTTP API

Install server dependencies and start the service:

```console
uv sync --extra server
uv run uvicorn mean_pic.server:app --host 0.0.0.0 --port 8000
```

The service provides `POST /encode`, `POST /decode`, `POST /interpolate`, and
`GET /health`. Configure it with `MEAN_PIC_MODEL_ID`, `MEAN_PIC_PROMPT`,
`MEAN_PIC_STEPS`, `MEAN_PIC_STRENGTH`, `MEAN_PIC_GUIDANCE_SCALE`,
`MEAN_PIC_DEVICE`, `MEAN_PIC_DTYPE`, and `MEAN_PIC_CPU_OFFLOAD=1`.

```console
uv run mean-pic cat.jpg fish.jpg result.png \
  --backend remote \
  --endpoint-url http://localhost:8000
```

If the environment variable named by `--api-token-env` is set, the client sends
it as a bearer token. The application itself does not authenticate requests;
terminate authentication in a trusted reverse proxy.

## Tests and examples

```console
uv run pytest
```

The tests use fake pipelines and do not download model weights. Diverse example
images and their licenses are documented in
[`tests/example/SOURCES.md`](tests/example/SOURCES.md).
