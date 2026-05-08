#!/usr/bin/env bash

# Parameters from notebook
output_dir="training_logs/3d-icon-sdxl-lora"
instance_prompt="3d icon in the style of TOK"
validation_prompt="a TOK icon of an astronaut riding a horse, in the style of TOK"
rank=8

accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="./data/3d_icon" \
  --instance_prompt="$instance_prompt" \
  --validation_prompt="$validation_prompt" \
  --output_dir="$output_dir" \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=3 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --adam_beta2=0.99 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --train_text_encoder_ti_frac=0.5\
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank="$rank" \
  --max_train_steps=1000 \
  --checkpointing_steps=2000 \
  --seed="0" \
  # --push_to_hub
