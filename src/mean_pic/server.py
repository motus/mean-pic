from __future__ import annotations

import os

from mean_pic.diffusion import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    DiffusionImageModel,
    DiffusionImageSettings,
)
from mean_pic.service import LatentImageService, create_app

MODEL_ID = os.environ.get("MEAN_PIC_MODEL_ID", DEFAULT_MODEL_ID)

model = DiffusionImageModel(
    DiffusionImageSettings(
        model_id=MODEL_ID,
        prompt=os.environ.get("MEAN_PIC_PROMPT", DEFAULT_PROMPT),
        max_denoising_steps=int(os.environ.get("MEAN_PIC_STEPS", "48")),
        strength=float(os.environ.get("MEAN_PIC_STRENGTH", "0.5")),
        guidance_scale=float(os.environ.get("MEAN_PIC_GUIDANCE_SCALE", "7.5")),
        device=os.environ.get("MEAN_PIC_DEVICE", "auto"),
        dtype=os.environ.get("MEAN_PIC_DTYPE", "auto"),
        cpu_offload=os.environ.get("MEAN_PIC_CPU_OFFLOAD", "0") == "1",
    )
)
app = create_app(LatentImageService(model, MODEL_ID))
