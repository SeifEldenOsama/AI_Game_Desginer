import os
import math
import torch
import itertools
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import login

from config import CFG
from model import AnimeHFDataset, load_models, apply_lora

def train():
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN", "YOUR_API")
    if hf_token != "YOUR_API":
        login(hf_token)
    else:
        print("Warning: HF_TOKEN not set. Using placeholder 'YOUR_API'.")

    # Load dataset
    print("Downloading anime dataset (streaming mode) ...")
    ds = load_dataset(CFG["hf_dataset"], split="train", streaming=True)
    ds = ds.shuffle(seed=CFG["seed"], buffer_size=1000)
    ds = list(itertools.islice(ds, CFG["max_samples"]))
    print(f"Dataset loaded: {len(ds)} images")

    img_col = next(
        (c for c in ds[0].keys() if "image" in c.lower() or "img" in c.lower()),
        list(ds[0].keys())[0]
    )
    print(f"Image column detected: '{img_col}'")

    # Load models
    print("Loading SD 1.5 components ...")
    tokenizer, text_enc, vae, unet, noise_sched = load_models(CFG["model_id"])
    unet = apply_lora(unet, CFG["lora_rank"])
    unet.print_trainable_parameters()

    # Prepare training
    torch.manual_seed(CFG["seed"])
    dataset    = AnimeHFDataset(ds, tokenizer, CFG["instance_prompt"], img_col, CFG["resolution"])
    dataloader = DataLoader(dataset, batch_size=CFG["train_batch"], shuffle=True, num_workers=2)

    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=CFG["lr"], weight_decay=1e-2
    )
    total_steps  = math.ceil(len(dataloader) / CFG["grad_accum"]) * CFG["num_epochs"]
    
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
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

if __name__ == "__main__":
    train()
