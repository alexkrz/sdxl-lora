import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()
pipe.load_lora_weights("checkpoints/emoji-lora/ios_emoji_xl_v2_lora.safetensors", lora_scale=0.6)

image = pipe("a rainbow unicorn icon emoji", negative_prompt="blurry").images[0]
image.save("results/emoji-lora.jpg")
