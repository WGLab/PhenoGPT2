import os
from datasets import load_dataset, Dataset
import torch
from tokenizers import AddedToken
from transformers import AutoModelForVision2Seq,AutoTokenizer, AutoProcessor, BitsAndBytesConfig 
import numpy as np
from peft import PeftModel, LoraConfig
from transformers import DataCollatorForSeq2Seq
import gc, json, pickle, itertools, argparse
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from datetime import datetime
import base64, re
from io import BytesIO
from PIL import Image
from trl import SFTConfig, SFTTrainer
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as F
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="PhenoGPT2 Image Recognizer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-train_data", "--train_data", required = True, help="directory to training dataset")
parser.add_argument("-eval_data", "--eval_data", required = True, help="directory to evaluation dataset")
parser.add_argument("-name", "--name", required = True, help="directory to output folder")
parser.add_argument("-styles", "--styles", required = False, action="store_true", help="Adding multiple styles to an image")
parser.add_argument("-lora", "--lora", required = False, action="store_true", help="LoRA finetuning")
parser.add_argument("-model_dir", "--model_dir", required = False, help="Directory to the Vision Foundation Model")
args = parser.parse_args()
if args.model_dir:
    model_id = args.model_dir
else:
    model_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
model = AutoModelForVision2Seq.from_pretrained(model_id,
    #quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map='auto')
processor = AutoProcessor.from_pretrained(model_id)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
processor.tokenizer.padding_side = "left"
model.language_model.resize_token_embeddings(len(processor.tokenizer)) ## go along with tokenizer.pad_token is None
model.config.pad_token_id = processor.tokenizer.pad_token_id
def get_last_checkpoint(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))
# ---------- reusable deterministic “base” preprocess --------------------------
# • keeps anatomy untouched
# • converts every file to RGB + a unified canvas size
#   (112×112 for your key data; bump to 224 if your backbone needs it)

def base_preprocess(img: Image.Image, target_size: int = 112) -> Image.Image:
    img = img.convert("RGB")
    # Make the smallest edge = target_size while keeping aspect ratio
    img.thumbnail((target_size, target_size), Image.Resampling.BICUBIC)
    # Put it on a square canvas so rotations don’t clip
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - img.width) // 2
    top  = (target_size - img.height) // 2
    new_img.paste(img, (left, top))
    return new_img


# ---------- light‑weight augmentations that respect dysmorphology -------------
# We package each style into its own callable transform for reproducibility.

def style_identity(img):
    return img  # original (always include)

def style_rot(img):
    # ±45° keeps landmarks visible
    angle = random.uniform(-45, 45)
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))

def style_hflip(img):
    # Beware of laterality! Use only if you have symmetrical phenotypes
    return ImageOps.mirror(img)

def style_color(img):
    # mild brightness / contrast / saturation jitter
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))
    return img

def style_noise(img):
    # subtle Gaussian blur + noise ( robustness to compression / focus )
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    noise = Image.effect_noise(img.size, random.randint(2,5))
    return Image.blend(img, noise.convert("RGB"), alpha=0.15)

STYLE_BANK = [style_identity, style_rot, style_hflip,
              style_color, style_noise]   # add / remove as you wish


# ---------- glue: make N variants from one corpus entry -----------------------

def generate_style_variants(sample: Dict, styles: bool) -> List[Dict]:
    """
    Args
    ----
    sample : one element of your corpus (contains 'img_path', 'output', etc.)
    n_styles : how many *additional* styles to generate (original is always kept)

    Returns
    -------
    List[Dict]  (length = n_styles + 1)  each ready for Llama‑3 Vision
    """

    def _wrap(pil_img, text_out):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": SYSTEM_MESSAGE + "\n" + USER_QUERY},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text_out}],
                },
            ],
            "image": pil_img,
        }

    SYSTEM_MESSAGE = (
        "You are a medical AI assistant with expertise in medical genetics, "
        "dysmorphology, and image-based phenotype recognition. Your task is to "
        "analyze facial and neck images to identify any observable phenotypes "
        "related to known human diseases. These phenotypes can include "
        "abnormalities in morphology, development, skin, face, hair, eyes, ears, "
        "nose, mouth, or neck. You must use the Human Phenotype Ontology (HPO) "
        "to normalize all detected phenotypes."
    )
    USER_QUERY = (
        "Analyze the attached facial image. "
        "Output only observable characteristics without implying any underlying "
        "cause, disease, or mechanism. All phenotypes should be separated by ||. "
        "Please do not provide any additional texts or formatting. "
        "The response should be like this: Phenotype A || Phenotype B || Phenotype C"
    )

    base_img = base_preprocess(Image.open(sample["img_path"]))
    variants = []
    if styles:
        transforms = STYLE_BANK.copy()
    else:
        transforms = STYLE_BANK[0]
    for t in STYLE_BANK:
        img_variant = t(base_img.copy())          # don’t mutate base
        variants.append(_wrap(img_variant, sample["output"]))

    return variants

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    images = [[e["image"]] for e in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    #batch = processor(text=texts, return_tensors="pt", padding=True)
    #batch = {k:v.to(device) for k,v in batch.items()}
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100

    batch["labels"] = labels
    
    return batch
def defining_args():
    return """
    meta-llama/Llama-3.2-11B-Vision-Instruct
    Pretraining on the HPO database
    Randomly initialize embeddings for new HPO tokens
    3 epochs
    """

def main():
    with open(args.train_data, 'rb') as f:
        train_input = pickle.load(f)
    with open(args.eval_data, 'rb') as f:
        eval_input = pickle.load(f)
    train_data = list(itertools.chain.from_iterable([generate_style_variants(t, args.styles) for t in train_input]))
    random.shuffle(train_data)
    eval_data = list(itertools.chain.from_iterable([generate_style_variants(t, args.styles) for t in eval_input]))
    random.shuffle(eval_data)    
    c = datetime.now()
    out_dir = os.getcwd() + f'./models/phenogpt2_L323BVision_{args.name}'
    os.makedirs(out_dir, exist_ok=True)
    out_dir_model = os.path.join(out_dir, 'model')
    os.makedirs(out_dir_model, exist_ok=True)
    with open(out_dir + '/params.txt', 'w') as f:
        f.write(defining_args())
    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,  # Reduce batch size to 2
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # Increase gradient accumulation steps
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=500,
        do_eval=True, ### SET this for pretraining
        eval_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=600,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=5000
    )


    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    if args.lora:
        LORA_R = 128 #128
        LORA_ALPHA = 256 #256
        LORA_DROPOUT= 0.05
        LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj","lm_head"]
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=collate_fn,
            dataset_text_field="", # needs dummy value
            peft_config=peft_config,
            tokenizer=processor.tokenizer,
        )
        os.makedirs(out_dir, exist_ok=True)
        out_dir_lora = os.path.join(out_dir, 'lora')
        trainer.train()
        trainer.save_model(out_dir_lora)
        #merged_model = model.merge_and_unload()
        #merged_model.save_pretrained(out_dir_model)
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=collate_fn,
            dataset_text_field="", # needs dummy value
            #peft_config=peft_config,
            tokenizer=processor.tokenizer,
        )
        trainer.train()
        trainer.save_model(out_dir_model)
    processor.save_pretrained(out_dir_model)
    processor.tokenizer.save_pretrained(out_dir_model)

    
    print(os.system("nvidia-smi"))

if __name__ == "__main__":
    main()
