import os
import gc
import json, re
import pickle
import argparse
from tqdm.auto import tqdm

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="PhenoGPT2 Qwen3 finetuning (HPO + demographics extraction) - DeepSpeed friendly"
    )

    parser.add_argument("--pretrain_model", type=str, required=True, help="Path to pretrained Qwen3 model dir")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data pickle file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data pickle file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-attn_implementation", "--attn_implementation", required=False, default='eager', help="Default implementation 'eager' is turned on by default. Note that: FlashAttention may not be supported on arm64/aarch64 platforms. Flash Attention helps faster inference and lower memory usage.")

    # DeepSpeed passthrough
    parser.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config json (e.g., ds_zero2_bf16.json)")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank for distributed training (added by DeepSpeed launcher)",
    )

    # (Optional) tokenize knobs
    parser.add_argument("--cutoff_len", type=int, default=13000, help="Max total tokens (prefix+answer)")
    parser.add_argument("--max_prefix_tokens", type=int, default=10000, help="If prefix exceeds, run clean_note()")

    return parser.parse_args()


# -------------------------
# Utility / prompt helpers
# -------------------------
def load_hpo_mapping(path="hpo_added_tokens.json"):
    with open(path, "r") as f:
        return json.load(f)


def form_json_output(data_point, name2hpo):
    phenotype_dict = {
        phen: {
            "HPO_ID": v["HPO_ID"],
            "onset": v["onset"],
        }
        for phen, v in data_point["output"]["phenotypes"].items()
        if phen in name2hpo
    }
    demographics = {k: v for k, v in data_point["output"]["demographics"].items() if k != "race"}
    return {"demographics": demographics, "phenotypes": phenotype_dict}


def generate_prompt(data_point, name2hpo):
    json_output = form_json_output(data_point, name2hpo)

    instruction = (
        "You are a genetic counselor specializing in extracting demographic details "
        "and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. "
        "Your task is to provide accurate and concise information without generating random answers. "
        "When demographic details or phenotype information is not explicitly mentioned in the input, "
        "use 'unknown' as the value."
    )

    question = (
        "Read the following input text and generate a JSON-formatted output with the following keys: "
        "demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, "
        "and ethnicity as keys. For the phenotype key, create a sub-dictionary where each HPO term is a key, "
        "and the value is a sub-dictionary that contains corresponding HPO identifier (HPO_ID) and the "
        "patient’s age (onset) when the phenotype first appeared, if mentioned in the text. "
        "If any information is unavailable, return 'unknown' for that field.\nInput: "
    )
    if 'clinical_note' in data_point:
        clinical_note = data_point["clinical_note"].replace("'", "").replace('"', "")
    else:
        clinical_note = data_point["input"].replace("'", "").replace('"', "")
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question + clinical_note},
        {"role": "assistant", "content": json.dumps(json_output)},
    ]
    return messages


LAB_VALUE_TOKEN = re.compile(
    r"""
    \b
    [A-Za-z][A-Za-z0-9()]*   # lab name (Na, HCO3, Creat, CK(CPK))
    -
    [-+]?\d+(\.\d+)?        # numeric value
    [*#]?                   # optional abnormal flag
    \b
    """,
    re.VERBOSE,
)

def is_lab_heavy_line(line, min_lab_tokens=3):
    tokens = LAB_VALUE_TOKEN.findall(line)
    return len(tokens) >= min_lab_tokens

def remove_inline_lab_blocks(text, min_block_lines=1):
    lines = text.splitlines()
    cleaned = []
    i = 0
    while i < len(lines):
        if is_lab_heavy_line(lines[i]):
            j = i
            while j < len(lines) and (is_lab_heavy_line(lines[j]) or lines[j].strip().startswith("___")):
                j += 1
            if j - i >= min_block_lines:
                i = j
                continue
        cleaned.append(lines[i])
        i += 1
    return "\n".join(cleaned)

DISCHARGE_TO_IM_END_RE = re.compile(
    r"(?is)"
    r"[ \t_]*discharge\s+medications\s*:"
    r".*?"
    r"(?=<\|im_end\|>)"
)
ADMISSION_TO_IM_END_RE = re.compile(
    r"(?is)"
    r"[ \t_]*medications\s+on\s+admission\s*:"
    r".*?"
    r"(?=<\|im_end\|>)"
)
def remove_discharge_meds_until_im_end(text: str) -> str:
    return DISCHARGE_TO_IM_END_RE.sub("", text)
def remove_admission_meds_until_im_end(text: str) -> str:
    return ADMISSION_TO_IM_END_RE.sub("", text)
def clean_note(note: str):
    note = remove_discharge_meds_until_im_end(note)
    note = remove_admission_meds_until_im_end(note)
    note = remove_inline_lab_blocks(note)
    return note


def tokenize_prompt(messages, tokenizer, cutoff_len=10000, max_prefix_tokens=10000):
    # ----- 1) Build PREFIX via chat template -----
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # ----- 2) Extract assistant content directly -----
    assert messages[-1]["role"] == "assistant"
    assistant_text = messages[-1]["content"]

    # ----- 3) Clean prefix if too long -----
    prefix_tmp = tokenizer(prefix_text, add_special_tokens=False)
    if len(prefix_tmp["input_ids"]) >= max_prefix_tokens:
        prefix_text = clean_note(prefix_text)

    # ----- 4) Tokenize separately (NO truncation) -----
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False, return_attention_mask=False)["input_ids"]

    # ----- 5) Ensure assistant EOS ONCE -----
    if not assistant_ids or assistant_ids[-1] != tokenizer.eos_token_id:
        assistant_ids.append(tokenizer.eos_token_id)

    # ----- 6) Enforce cutoff (truncate PREFIX ONLY) -----
    total_len = len(prefix_ids) + len(assistant_ids)
    if total_len > cutoff_len:
        keep_prefix = cutoff_len - len(assistant_ids)
        if keep_prefix <= 0:
            raise ValueError("Assistant output alone exceeds cutoff_len — cannot truncate safely.")
        prefix_ids = prefix_ids[-keep_prefix:]

    # ----- 7) Assemble -----
    input_ids = prefix_ids + assistant_ids
    attention_mask = [1] * len(input_ids)

    # ----- 8) Labels: mask prefix -----
    labels = [-100] * len(prefix_ids) + assistant_ids.copy()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def generate_and_tokenize(data_point, tokenizer, name2hpo, cutoff_len, max_prefix_tokens):
    messages = generate_prompt(data_point, name2hpo)
    return tokenize_prompt(messages, tokenizer, cutoff_len=cutoff_len, max_prefix_tokens=max_prefix_tokens)


# -------------------------
# Collator (safe for causal LM + custom labels)
# -------------------------
def make_causal_lm_collator(tokenizer, pad_to_multiple_of=8):
    pad_id = tokenizer.pad_token_id

    def _collate(features):
        # length to pad to (optionally multiple-of for Tensor Cores)
        max_len = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        def pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor([pad(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor([pad(f.get("attention_mask", [1]*len(f["input_ids"])), 0) for f in features], dtype=torch.long)
        labels = torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return _collate


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    gc.collect()
    torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    model_out = os.path.join(args.output_dir, "model")
    tokenizer_out = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(model_out, exist_ok=True)
    os.makedirs(tokenizer_out, exist_ok=True)

    # Save run config
    with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Debug DS env
    print(
        f"[RANK DEBUG] MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
        f"RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}",
        flush=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model, use_fast=True)

    # Ensure pad token exists (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    # Load HPO mapping
    name2hpo = load_hpo_mapping()

    # Load model (IMPORTANT: no device_map="auto" for DeepSpeed)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,  # keep if your env supports it
        # trust_remote_code=True/False depending on your Qwen3 setup
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    ## set LLaMA chat template
    config = model.config
    if config.model_type == 'llama':
        tokenizer.chat_template = tokenizer.chat_template = """{% for message in messages %}
    {% if message['role'] == 'system' %}
    <|start_header_id|>system<|end_header_id|>
    {{ message['content'] }}<|eot_id|>
    {% elif message['role'] == 'user' %}
    <|start_header_id|>user<|end_header_id|>
    {{ message['content'] }}<|eot_id|>
    {% elif message['role'] == 'assistant' %}
    <|start_header_id|>assistant<|end_header_id|>
    {{ message['content'] }}<|eot_id|>
    {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}
    <|start_header_id|>assistant<|end_header_id|>
    {% endif %}
    """

    # Load data
    with open(args.train_data, "rb") as f:
        train_data_raw = pickle.load(f)
    with open(args.val_data, "rb") as f:
        val_data_raw = pickle.load(f)

    train_data_processed = list(train_data_raw.values()) if isinstance(train_data_raw, dict) else train_data_raw
    val_data_processed = list(val_data_raw.values()) if isinstance(val_data_raw, dict) else val_data_raw

    print("Example prompt:\n", generate_prompt(train_data_processed[0], name2hpo), flush=True)

    # Tokenize (simple eager version, like your original)
    train_data = [
        generate_and_tokenize(dp, tokenizer, name2hpo, args.cutoff_len, args.max_prefix_tokens)
        for dp in tqdm(train_data_processed, desc="Tokenizing train")
    ]
    val_data = [
        generate_and_tokenize(dp, tokenizer, name2hpo, args.cutoff_len, args.max_prefix_tokens)
        for dp in tqdm(val_data_processed, desc="Tokenizing val")
    ]

    train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
    val_dataset = Dataset.from_dict({k: [d[k] for d in val_data] for k in val_data[0]})

    data_collator = make_causal_lm_collator(tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        dataloader_num_workers=4,

        bf16=True,
        tf32=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        optim="adamw_torch_fused",

        logging_strategy="steps",
        logging_steps=1000,

        save_strategy="steps",
        save_steps=10000,
        save_total_limit=2,

        do_eval=True,
        eval_strategy="steps",
        eval_steps=1000,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=4,

        num_train_epochs=10,
        warmup_ratio=0.03,
        weight_decay=0.05,
        learning_rate=1e-5,

        # IMPORTANT: only meaningful for DDP; DeepSpeed handles this too, but it's fine to keep:
        ddp_find_unused_parameters=False,

        push_to_hub=False,

        # NEW: enable DeepSpeed if provided
        deepspeed=args.deepspeed,
        # Optional: helps long runs resume cleanly
        # save_safetensors=True,
        # report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # quick sanity batch
    for batch in trainer.get_train_dataloader():
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print("Input (decoded):", input_text[:500], "...\n", flush=True)

        if (labels >= 151669).sum() > 0:
            print("✅ Found >=151669 in labels!", flush=True)
        else:
            print("❌ Missing >=151669 in labels!", flush=True)
        break

    try:
        trainer.train()
        trainer.save_model(model_out)
        tokenizer.save_pretrained(model_out)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        os.system("nvidia-smi")

if __name__ == "__main__":
    main()
