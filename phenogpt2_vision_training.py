import os
import gc
import pickle, json
import random
import argparse
from typing import List, Dict, Any

from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a medical AI assistant specialized in dysmorphology and phenotype extraction from clinical facial photographs.

Your task is to analyze a single patient image and extract only directly observable phenotypic findings.

Rules:
- Mention only visible phenotypes.
- Do not infer diseases, syndromes, genes, mechanisms, ethnicity, age, or history unless directly visible and clinically obvious from the image.
- Do not speculate.
- Use concise phenotype phrases that are easy to normalize to Human Phenotype Ontology (HPO).
- Avoid explanations, reasoning, confidence statements, or extra prose.
- If the image quality is poor, still extract only the findings that are reasonably visible.
- If no clear phenotype can be reliably identified, output exactly: No clearly observable phenotype"""

USER_PROMPT = """Analyze the attached patient image and return only the observable phenotype list.

Output format:
Phenotype A || Phenotype B || Phenotype C

Strict requirements:
- Output only phenotype phrases separated by ' || '.
- No bullet points.
- No numbering.
- No diagnosis.
- No explanation.
- No extra text."""

# with open('/home/nguyenqm/projects/collaboration/aaron/mai/visual_hpo_terms.json', 'r') as f:
#     visual_hpos = json.load(f)
# ---------------------------------------------------------
# Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-train_data", "--train_data", required=True)
parser.add_argument("-eval_data", "--eval_data", required=True)
parser.add_argument("-model", "--model", default="Qwen/Qwen3-VL-8B-Instruct")
parser.add_argument("-name", "--name", required=True)
parser.add_argument("-styles", "--styles", action="store_true")
parser.add_argument("-lora", "--lora", action="store_true")
args = parser.parse_args()


DEFAULTS = {
    "rotation_deg": 20.0,
    "brightness_jitter": 0.06,
    "contrast_jitter": 0.06,
    "color_jitter": 0.03,
    "blur_prob": 0.08,
    "max_len": 1500,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
}
def load_image(path):
    return Image.open(path).convert("RGB")


def conservative_augment(img):
    img = img.rotate(random.uniform(-DEFAULTS["rotation_deg"], DEFAULTS["rotation_deg"]))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.94, 1.06))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.94, 1.06))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.97, 1.03))
    if random.random() < DEFAULTS["blur_prob"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def build_messages(text, model_family):
    # QWEN (UNCHANGED)
    if model_family == "qwen":
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]

    # LLAMA (MINIMAL CHANGE)
    elif model_family == "llama":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT}
            ]},
            {"role": "assistant", "content": text},
        ]

    # fallback
    else:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": text},
        ]
def generate_style_variants(sample, styles):
    image_path = sample["image_path"]
    text = sample["output"]

    variants = [{"image_path": image_path, "answer_text": text, "use_aug": False}]
    if styles:
        for _ in range(2):
            variants.append({"image_path": image_path, "answer_text": text, "use_aug": True})
    return variants


def load_pickle(path):
    return pickle.load(open(path, "rb"))


def is_arrow_dataset(path):
    return os.path.isdir(path)


def load_arrow_as_legacy(path, data_type='train'):
    ds = load_from_disk(path)[data_type]
    converted = []

    for row in ds:
        img = row["image_restored"]
        tmp_path = f"/tmp/arrow_{random.randint(0, 1_000_000_000_000)}.png"

        if isinstance(img, Image.Image):
            img.save(tmp_path)
        else:
            Image.fromarray(img).save(tmp_path)

        hpo = row["hpo_terms"]
        if not hpo:
            text = "No phenotype detected"
        else:
            text = " | ".join([str(x).lower() for x in hpo])

        converted.append({
            "image_path": tmp_path,
            "output": text
        })

    return converted


class PhenotypeDataset(Dataset):
    def __init__(self, items, model_family):
        self.items = items
        self.model_family = model_family

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = load_image(item["image_path"])

        if item.get("use_aug", False):
            img = conservative_augment(img)

        return {
            "messages": build_messages(item["answer_text"], self.model_family),
            "image": img
        }

def collate_fn(processor, model_family):
    def fn(batch: List[Dict[str, Any]]):
        full_texts = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in batch
        ]

        prompt_texts = [
            processor.apply_chat_template(
                ex["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]

        if model_family == "llama":
            images = [[ex["image"]] for ex in batch]  # 🔥 nested list
        else:
            images = [ex["image"] for ex in batch]   # keep qwen unchanged

        enc = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=DEFAULTS["max_len"],
        )

        input_ids = enc["input_ids"]
        pad_id = processor.tokenizer.pad_token_id

        prompt_ids = processor.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=DEFAULTS["max_len"],
            add_special_tokens=False,
        )["input_ids"]

        prompt_lens = (prompt_ids != pad_id).sum(dim=1)

        labels = input_ids.clone()
        bs, seqlen = labels.shape

        for i in range(bs):
            pl = int(prompt_lens[i].item())
            pl = min(pl, seqlen)
            labels[i, :pl] = -100

        labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc

    return fn


def load_model():
    quant = BitsAndBytesConfig(load_in_8bit=True) if args.lora else None
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    print(f"DISTRIBUTED MODE: {distributed}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        trust_remote_code=False,
        attn_implementation="sdpa" if "llama" in args.model.lower() else "flash_attention_2"
    )

    # if distributed:
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         args.model,
    #         **model_kwargs,
    #     )
    # else:
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        device_map="auto" if not args.lora else None,
        **model_kwargs,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)

    # if processor.tokenizer.pad_token is None:
    #     processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({
            "pad_token": processor.tokenizer.eos_token
        })
    model.config.use_cache = False
    return model, processor


def build_peft_config():
    return LoraConfig(
        r=DEFAULTS["lora_r"],
        lora_alpha=DEFAULTS["lora_alpha"],
        target_modules=DEFAULTS["lora_target_modules"],
        lora_dropout=DEFAULTS["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
def get_model_family(model_name: str):
    if "qwen" in model_name.lower():
        return "qwen"
    elif "llama" in model_name.lower():
        return "llama"
    else:
        return "generic"

def main():
    set_seed(42)
    # print(
    #     f"[RANK DEBUG] MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
    #     f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
    #     f"RANK={os.environ.get('RANK')} "
    #     f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
    #     f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}",
    #     flush=True
    # )

    print("Loading the data")
    if is_arrow_dataset(args.train_data):
        train_input = load_arrow_as_legacy(args.train_data, 'train')
        eval_input = load_arrow_as_legacy(args.eval_data, 'validation')
    else:
        train_input = load_pickle(args.train_data)
        eval_input = load_pickle(args.eval_data)

    train_items = []
    for t in train_input:
        train_items.extend(generate_style_variants(t, args.styles))

    eval_items = []
    for t in eval_input:
        eval_items.extend(generate_style_variants(t, False))

    print("Formatting the data")
    model_family = get_model_family(args.model)

    train_dataset = PhenotypeDataset(train_items, model_family)
    eval_dataset = PhenotypeDataset(eval_items, model_family)

    print("Loading the model")
    model, processor = load_model()
    if "llama" in args.model.lower():
        processor.tokenizer.padding_side = "right"
    print("Setting up the training configs")
    training_args = SFTConfig(
        output_dir=f"./vision_{args.name}",
        num_train_epochs=5,
        per_device_train_batch_size=4 if "llama" in model_family.lower() else 16,
        gradient_accumulation_steps=10 if "llama" in model_family.lower() else 8,
        learning_rate=2e-4 if args.lora else 1e-5,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        warmup_steps=10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    peft_config = build_peft_config() if args.lora else None

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn(processor, model_family),
        peft_config=peft_config,
    )

    print("Start to train the model")
    try:
        trainer.train()
        trainer.save_model(f"./vision_{args.name}/model")
        processor.save_pretrained(f"./vision_{args.name}/model")
        processor.tokenizer.save_pretrained(f"./vision_{args.name}/model")
    except Exception as e:
        print(f"Error: {e}")
        os.system("nvidia-smi")
    os.system("nvidia-smi")
    print("Done")


if __name__ == "__main__":
    main()
