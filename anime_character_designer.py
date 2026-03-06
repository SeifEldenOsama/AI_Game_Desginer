!pip install -q -U \
    diffusers==0.27.2 \
    transformers accelerate safetensors \
    peft datasets gradio \
    Pillow torchvision

!pip uninstall -y websockets fsspec

!pip install -q \
    "diffusers==0.30.3" \
    "huggingface_hub==0.23.4" \
    "transformers==4.41.2" \
    accelerate \
    safetensors \
    peft \
    datasets \
    gradio \
    Pillow \
    torchvision

import os
os.kill(os.getpid(), 9)

import os, math, gc, torch
from pathlib import Path
from PIL import Image
import gradio as gr

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from datasets import load_dataset

assert torch.cuda.is_available(), "CUDA is not available !"
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

CFG = dict(
    model_id         = "runwayml/stable-diffusion-v1-5",
    output_dir       = "/content/lora_anime",

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
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

from huggingface_hub import login
login("YOUR_API")

from datasets import load_dataset, DownloadConfig
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

download_config = DownloadConfig(max_retries=10)

from datasets import load_dataset
import itertools

print("Downloading anime dataset (streaming mode) ...")

ds = load_dataset(
    CFG["hf_dataset"],
    split="train",
    streaming=True
)


ds = ds.shuffle(
    seed=CFG["seed"],
    buffer_size=1000
)


ds = itertools.islice(ds, CFG["max_samples"])


ds = list(ds)

print(f"Dataset loaded: {len(ds)} images")


img_col = next(
    (c for c in ds[0].keys() if "image" in c.lower() or "img" in c.lower()),
    list(ds[0].keys())[0]
)

print(f"Image column detected: '{img_col}'")

class AnimeHFDataset(Dataset):
    def __init__(self, hf_ds, tokenizer, prompt, img_col, size=512):
        self.ds        = hf_ds
        self.tokenizer = tokenizer
        self.prompt    = prompt
        self.img_col   = img_col
        self.tfm = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img  = item[self.img_col]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        pixel = self.tfm(img)
        ids   = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel, "input_ids": ids}
print("Dataset Class Defined")

print("Loading SD 1.5 components ...")
tokenizer   = CLIPTokenizer.from_pretrained(CFG["model_id"], subfolder="tokenizer")
text_enc    = CLIPTextModel.from_pretrained(CFG["model_id"], subfolder="text_encoder")
vae         = AutoencoderKL.from_pretrained(CFG["model_id"], subfolder="vae")
unet        = UNet2DConditionModel.from_pretrained(CFG["model_id"], subfolder="unet")
noise_sched = DDPMScheduler.from_pretrained(CFG["model_id"], subfolder="scheduler")

vae.requires_grad_(False)
text_enc.requires_grad_(False)
print("Model loaded successfully")

from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r             = CFG["lora_rank"],
    lora_alpha    = CFG["lora_rank"],
    target_modules= ["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout  = 0.05,
    bias          = "none",
)
unet = get_peft_model(unet, lora_cfg)
unet.print_trainable_parameters()

import math
from diffusers.optimization import get_scheduler as get_lr_scheduler
from accelerate import Accelerator

torch.manual_seed(CFG["seed"])

dataset    = AnimeHFDataset(ds, tokenizer, CFG["instance_prompt"], img_col, CFG["resolution"])
dataloader = DataLoader(dataset, batch_size=CFG["train_batch"], shuffle=True, num_workers=2)

optimizer  = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=CFG["lr"], weight_decay=1e-2
)
total_steps  = math.ceil(len(dataloader) / CFG["grad_accum"]) * CFG["num_epochs"]
lr_scheduler = get_lr_scheduler(
    "cosine", optimizer=optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)

accelerator = Accelerator(
    mixed_precision=CFG["mixed_precision"],
    gradient_accumulation_steps=CFG["grad_accum"],
)
unet, text_enc, vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    unet, text_enc, vae, optimizer, dataloader, lr_scheduler
)

dtype = torch.float16
print(f"\nTraining started | {CFG['num_epochs']} epochs | {total_steps} total steps\n")

for epoch in range(CFG["num_epochs"]):
    unet.train()
    epoch_loss = 0.0

    for batch in dataloader:
        with accelerator.accumulate(unet):
            latents    = vae.encode(batch["pixel_values"].float()).latent_dist.sample().to(dtype) * 0.18215
            noise      = torch.randn_like(latents)
            timesteps  = torch.randint(0, noise_sched.config.num_train_timesteps,
                                       (latents.shape[0],), device=latents.device).long()
            noisy_lat  = noise_sched.add_noise(latents, noise, timesteps)
            enc_hidden = text_enc(batch["input_ids"])[0]
            pred       = unet(noisy_lat, timesteps, enc_hidden).sample
            loss       = torch.nn.functional.mse_loss(pred.float(), noise.float())
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach().item()

    avg = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1:3d}/{CFG['num_epochs']}]  loss: {avg:.4f}  lr: {lr_scheduler.get_last_lr()[0]:.2e}")

    if (epoch + 1) % CFG["save_every"] == 0:
        ckpt = f"{CFG['output_dir']}/epoch_{epoch+1}"
        accelerator.unwrap_model(unet).save_pretrained(ckpt)
        print(f"  Checkpoint saved -> {ckpt}")

accelerator.unwrap_model(unet).save_pretrained(f"{CFG['output_dir']}/final")
print("\nFine-tuning complete!")

import gc
from peft import PeftModel
from diffusers import StableDiffusionPipeline

del optimizer, dataloader, lr_scheduler, accelerator
gc.collect()
torch.cuda.empty_cache()

base_unet = UNet2DConditionModel.from_pretrained(
    CFG["model_id"], subfolder="unet", torch_dtype=torch.float16
)
ft_unet = PeftModel.from_pretrained(base_unet, f"{CFG['output_dir']}/final")
ft_unet = ft_unet.merge_and_unload()

pipe = StableDiffusionPipeline.from_pretrained(
    CFG["model_id"], unet=ft_unet, torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print("Fine-tuned pipeline is ready!")

!pip install --upgrade gradio gradio_client

import gradio as gr

EXAMPLES = [
    ["warrior with red armor and katana",    "blurry, low quality", 7.5, 30, 42],
    ["mage girl with purple magic aura",     "blurry, low quality", 7.5, 30, 0],
    ["cyberpunk ninja with neon highlights", "blurry, low quality", 8.0, 35, 7],
]

def generate(prompt, neg, guidance, steps, seed):
    full_prompt = f"anime character, sks style, {prompt}, high quality, detailed"
    generator   = None if int(seed) == 0 else torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(
        prompt              = full_prompt,
        negative_prompt     = neg,
        guidance_scale      = float(guidance),
        num_inference_steps = int(steps),
        generator           = generator,
    ).images[0]
    return image

ui = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Character Description", value="warrior with golden armor"),
        gr.Textbox(label="Negative Prompt",       value="blurry, low quality, deformed"),
        gr.Slider(1, 15,  value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=30,  step=5,   label="Inference Steps"),
        gr.Number(value=0, label="Seed (0 = random)"),
    ],
    outputs=gr.Image(label="Generated Anime Character"),
    examples=EXAMPLES,
    title="🎌 Anime Game Character Designer — Fine-Tuned",
    description=(
        "Stable Diffusion 1.5 fine-tuned with LoRA on anime characters.\n"
        "Trigger word: anime character, sks style"
    ),
)
ui.launch(share=True)
