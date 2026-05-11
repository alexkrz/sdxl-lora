#!/usr/bin/env bash

# Parameters from notebook
output_dir="training_logs/3d-icon-sdxl-lora-v6"  # Need to verify that slashes work correctly in output_dir
instance_prompt="3d icon in the style of TOK"
validation_prompt="a TOK icon of an astronaut riding a horse, in the style of TOK"
rank=8

accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="./data/3d_icon" \
  --instance_prompt="$instance_prompt" \
  --validation_prompt="$validation_prompt" \
  --num_validation_images=4 \
  --validation_epochs=30 \
  --output_dir="$output_dir" \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=2 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --optimizer="AdamW"\
  --learning_rate=1e-4 \
  --text_encoder_lr=5e-5 \
  --adam_beta1=0.9 \
  --adam_beta2=0.99 \
  --adam_weight_decay=0.01 \
  --train_text_encoder_ti\
  --train_text_encoder_ti_frac=0.5\
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank="$rank" \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --seed=42 \
  # --push_to_hub
