# %%
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file, save_file


def build_fixed_lora_file(weight_path: Path) -> Path | None:
    """Create a sibling *_fixed.safetensors file for legacy SDXL LoRA checkpoints."""
    state_dict = load_file(str(weight_path))

    # Legacy checkpoints can contain keys like "unet.up_blocks..." which PEFT expects without
    # the "unet." prefix for target module matching.
    needs_repair = any(k.startswith("unet.") and "lora" in k for k in state_dict.keys())
    if not needs_repair:
        return None

    repaired_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("unet.") and ("lora" in key or key.endswith(".alpha")):
            new_key = key[len("unet.") :]
        else:
            new_key = key

        if new_key in repaired_state_dict:
            raise ValueError(f"Key collision while repairing LoRA file: {new_key}")
        repaired_state_dict[new_key] = value

    fixed_path = weight_path.with_name(f"{weight_path.stem}_fixed{weight_path.suffix}")
    save_file(repaired_state_dict, str(fixed_path))
    return fixed_path


pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Load lora weights from local checkpoint
checkpoint_dir = Path("checkpoints/3d-icon-sdxl-lora")
weight_name = "pytorch_lora_weights.safetensors"
weight_path = checkpoint_dir / weight_name
fixed_weight_path = checkpoint_dir / f"{weight_path.stem}_fixed{weight_path.suffix}"

try:
    pipe.load_lora_weights(str(checkpoint_dir), weight_name=weight_name)
except ValueError as exc:
    message = str(exc)
    if "Target modules" in message and "not found in the base model" in message:
        if not fixed_weight_path.exists():
            build_fixed_lora_file(weight_path)
        if fixed_weight_path.exists():
            pipe.load_lora_weights(str(checkpoint_dir), weight_name=fixed_weight_path.name)
        else:
            raise
    else:
        raise
