# SDXL LoRA

Train a SDXL LoRA adapter as described here: <https://huggingface.co/blog/sdxl_lora_advanced_script>

## Setup

I believe the future of python environment management will be `uv`.
Currently, I prefer to install the requirements in a `conda` virtual environment.
Therefore, I suggest to set up your environment with the following commands:

```bash
conda create -n $YOUR_ENV_NAME python=3.12
conda activate $YOUR_ENV_NAME
uv pip install -r requirements.txt
```

Alternatively, you can also use `uv` to create a virtual environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training

As in the accompanying Colab notebook from the blogpost mentioned on top, we base our model finetuning on the `train_dreambooth_lora_sdxl_advanced.py` script from the [diffusers](https://pypi.org/project/diffusers/) library.
The script allows us to combine the concepts of
[Low Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685),
[Dreambooth](https://arxiv.org/abs/2208.12242) and
[Textual Inversion](https://arxiv.org/abs/2208.01618).

For successfully finetuning a Stable Diffusion model, an appropriate choice of training hyperparameters is very important.
We have gathered a set of training arguments in the [train.sh](./train.sh) shell script.
The script makes use of the [accelerate](https://pypi.org/project/accelerate/) library to perform training.

Training can be started by simply running

```bash
bash train.sh
```

## Inference

Once you have trained your own LoRA adapter or if you want to load a LoRA adapter from the Huggingface Hub, you have to load it into the Stable Diffusion pipeline.
We demonstrate how to do so in the final part of the [sdxl_dreambooth_lora.ipynb](./sdxl_dreambooth_lora.ipynb) notebook.

The code for loading the LoRA adapter and the Textual Inversion embeddings and then generating your own images looks as follows:

```python
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo_id = f"$PLACEHOLDER"

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Load lora weights from checkpoint
pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")

# Load embeddings for textual inversion
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(repo_id=repo_id, filename="ti_embeddings.safetensors", repo_type="model")

state_dict = load_file(embedding_path)
# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (CLIP ViT-G/14)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

instance_token = "<s0><s1>"
prompt = f"a {instance_token} icon of an orange llama eating ramen, in the style of {instance_token}"

image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
image
```
