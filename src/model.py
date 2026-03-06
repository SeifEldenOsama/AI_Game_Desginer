import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

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

def load_models(model_id):
    tokenizer   = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_enc    = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae         = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet        = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    
    return tokenizer, text_enc, vae, unet, noise_sched

def apply_lora(unet, lora_rank):
    lora_cfg = LoraConfig(
        r             = lora_rank,
        lora_alpha    = lora_rank,
        target_modules= ["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout  = 0.05,
        bias          = "none",
    )
    unet = get_peft_model(unet, lora_cfg)
    return unet
