import os
from pathlib import Path

CFG = dict(
    model_id         = "runwayml/stable-diffusion-v1-5",
    output_dir       = "./lora_anime",

    hf_dataset       = "huggan/anime-faces",
    hf_image_col     = "image",
    max_samples      = 200,

    instance_prompt  = "anime character, sks style",
    neg_prompt       = "blurry, low quality, deformed, watermark, text",

    resolution       = 512,
    train_batch      = 1,
    grad_accum       = 4,
    lr               = 1e-4,
    num_epochs       = 20,
    save_every       = 10,
    lora_rank        = 8,
    mixed_precision  = "fp16",
    seed             = 42,
)

# Ensure output directory exists
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
