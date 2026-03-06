import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
from config import CFG

def load_pipeline():
    print("Loading fine-tuned pipeline ...")
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
    return pipe

pipe = None

def generate(prompt, neg, guidance, steps, seed):
    global pipe
    if pipe is None:
        pipe = load_pipeline()
        
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

EXAMPLES = [
    ["warrior with red armor and katana",    "blurry, low quality", 7.5, 30, 42],
    ["mage girl with purple magic aura",     "blurry, low quality", 7.5, 30, 0],
    ["cyberpunk ninja with neon highlights", "blurry, low quality", 8.0, 35, 7],
]

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

if __name__ == "__main__":
    ui.launch(share=True)
