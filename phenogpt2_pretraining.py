import os
from datasets import load_dataset, Dataset#, load_from_disk
import datasets
import torch
from tokenizers import AddedToken, pre_tokenizers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import DataCollatorForSeq2Seq
torch.backends.cuda.matmul.allow_tf32 = True
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm.auto import tqdm
import gc, json, pickle, joblib, argparse
gc.collect()
torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_prompt(data_point):
    instruction = "You are a genetic counselor and you are learning everything about Human Phenotype Ontology (HPO) database."
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": data_point['user']},
        {"role": "assistant", "content": data_point['assistant']}
        ]
    return messages
def tokenize_prompt(messages, tokenizer, cutoff_len=2000):
    # ----- 1) Build prefix (system + user) -----
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # ----- 2) Build full conversation (includes assistant answer) -----
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # ----- 4) Tokenize prefix -----
    prefix = tokenizer(
        prefix_text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=False,
    )

    # ----- 5) Tokenize full -----
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # ----- 6) Labels: mask prefix tokens -----
    labels = [-100] * len(input_ids)
    prefix_len = len(prefix["input_ids"])
    labels[prefix_len:] = input_ids[prefix_len:]

    # ----- 7) Ensure EOS -----
    if input_ids[-1] != tokenizer.eos_token_id and len(input_ids) < cutoff_len:
        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)
        labels.append(tokenizer.eos_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def generate_and_tokenize(data_point, tokenizer):
    messages = generate_prompt(data_point)
    return tokenize_prompt(messages, tokenizer)
def defining_args():
    return """
    Qwen/Qwen3-8B
    pretraining on the HPO database
    /home/nguyenqm/projects/PhenoGPT2/new_synthetic_data/pretraining_hpodb.pkl
    Randomly initialize the embeddings 
    embedding_layer = model.get_input_embeddings()
    new_vocab_size = embedding_layer.num_embeddings
    embedding_dim = embedding_layer.embedding_dim

    num_new_tokens = new_vocab_size - original_vocab_size

    if num_new_tokens > 0:
        with torch.no_grad():
            # Generate random embeddings (normal distribution)
            new_weights = torch.empty((num_new_tokens, embedding_dim), dtype=embedding_layer.weight.dtype)
            torch.nn.init.normal_(new_weights, mean=0.0, std=embedding_dim ** -0.5)

            # Assign to new rows
            embedding_layer.weight[original_vocab_size:] = new_weights
    3 epochs
    WE NEED TO TOKENIZE THE DATA '''AFTER''' ADDING THE NEW TOKENS. OTHERWISE, THEY LEARN THE NORMAL WAYS.
    No validations
    """
def main():
    """
    Set training parameters and train model
    """
    parser = argparse.ArgumentParser(description="PhenoGPT2 HPO Aware Pretrain Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-train_data", "--train_data", required = True, help="directory to training dataset")
    parser.add_argument("-name", "--name", required = True, help="directory to output folder")
    parser.add_argument("-lora", "--lora", required = False, action="store_true", help="LoRA finetuning")
    parser.add_argument("-model_dir", "--model_dir", required = False, help="Directory to the Foundation Model")
    parser.add_argument("-attn_implementation", "--attn_implementation", required=False, default='eager', help="Default implementation 'eager' is turned on by default. Note that: FlashAttention may not be supported on arm64/aarch64 platforms. Flash Attention helps faster inference and lower memory usage.")
    args = parser.parse_args()
    if args.model_dir:
        model_name = args.model_dir
    else:
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    print('Start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    c= datetime.now()
    if args.name:
        out_dir_model = os.getcwd() + '/models/' + args.name
    else:
        out_dir_model = os.getcwd() + '/models/hpo_aware_pretrain'
    os.makedirs(out_dir_model, exist_ok=True)
    # with open(out_dir_model + '/params.txt', 'w') as f:
    #     f.write(defining_args())
    print(out_dir_model)
    original_vocab_size = len(tokenizer)
    print('Start loading model')
    model=AutoModelForCausalLM.from_pretrained(model_name,do_sample=True, #quantization_config=quantization_config,
                                            attn_implementation=args.attn_implementation,  # keep if your env supports it
                                            torch_dtype=torch.bfloat16, device_map = 'auto')
    with open('./data/hpo_added_tokens.json', 'r') as f:
        name2hpo = json.load(f)
    all_hpo_ids = list(np.unique(list(name2hpo.values())))
    for hpo_id in tqdm(all_hpo_ids, desc = 'Adding Tokens'):
        tokenizer.add_tokens([AddedToken(hpo_id, single_word=True, normalized=False,special=False)], special_tokens=False)
    model.resize_token_embeddings(len(tokenizer)) ## go along with tokenizer.pad_token is None
    model.config.pad_token_id = tokenizer.pad_token_id
    embedding_layer = model.get_input_embeddings()
    new_vocab_size = embedding_layer.num_embeddings
    embedding_dim = embedding_layer.embedding_dim

    num_new_tokens = new_vocab_size - original_vocab_size

    if num_new_tokens > 0:
        with torch.no_grad():
            # Generate random embeddings (normal distribution)
            new_weights = torch.empty((num_new_tokens, embedding_dim), dtype=embedding_layer.weight.dtype)
            torch.nn.init.normal_(new_weights, mean=0.0, std=embedding_dim ** -0.5)

            # Assign to new rows
            embedding_layer.weight[original_vocab_size:] = new_weights
    print('Processing data')
    with open(args.train_data, 'rb') as f: # REPLACE YOUR TRAINING DATA HERE
        train_data = pickle.load(f)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print(generate_prompt(train_data[0]))
    train_data = list(map(lambda x: generate_and_tokenize_prompt(x, tokenizer), tqdm(train_data, desc="Processing train")))
    val_data = list(map(lambda x: generate_and_tokenize_prompt(x, tokenizer), tqdm(val_data, desc="Processing val")))
    train_data = {key: [item[key] for item in train_data] for key in train_data[0]}
    train_data = Dataset.from_dict(train_data)
    val_data = {key: [item[key] for item in val_data] for key in val_data[0]}
    val_data = Dataset.from_dict(val_data)
    print("Setting training arguments")
    
    training_args = TrainingArguments(
        output_dir=out_dir_model,
        label_names=["labels"],
        # =================== PERFORMANCE ===================
        per_device_train_batch_size=32,  # H100 can handle this
        gradient_accumulation_steps=4,   # Total batch size: 32 * 4 * 8 = 1024
        dataloader_num_workers=4,
        fp16=False,  # Using bf16 instead
        bf16=True,   # Preferred on H100 for better throughput and stability
        tf32=True,
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="adamw_torch_fused",  # Fused AdamW performs well on H100
        
        # =================== LOGGING ===================
        logging_dir=f"{out_dir_model}/logs",
        logging_strategy="steps",
        logging_steps=1000,  # More frequent for better monitoring
        
        # =================== SAVING ===================
        save_strategy="steps",
        save_steps=1000,  # More frequent saves, faster recovery
        save_total_limit=2,  # Prevent disk bloating
        
        # =================== EVALUATION ===================
        do_eval=False,
        # eval_strategy="steps",
        # eval_steps=10000,
        
        # =================== TRAINING CONTROL ===================
        num_train_epochs=3,  # ~3 epochs likely enough for new-token adaptation
        max_steps=-1,        # Let epochs control the length
        
        warmup_ratio=0.03,  # ~3–5% is enough
        weight_decay=0.05,  # Slightly stronger regularization

        # =================== DISTRIBUTED ===================
        ddp_find_unused_parameters=False,  # Needed for gradient checkpointing + DDP

        # =================== MISC ===================
        push_to_hub=False,
        resume_from_checkpoint=True  # Auto-resume
    )
    if args.lora:
        LORA_R = 128 #128
        LORA_ALPHA = 256 #256
        LORA_DROPOUT= 0.05
        LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj","lm_head"]
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, config)
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        #eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    for batch in trainer.get_train_dataloader():
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        # Decode first sample in the batch
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        #label_text = tokenizer.decode(labels[0], skip_special_tokens=True)

        print("Input:", input_text)
    #print("Label:", label_text)
        if (labels >= 151669).sum() > 0:
            print("✅ Found in labels!")
        else:
            print("❌ Missing in labels!")
        break
    try:
        trainer.train()
        trainer.save_model(out_dir_model) #save adapter
        tokenizer.save_pretrained(out_dir_model)
        # 2. Merge adapters into base model
        #model = trainer.model.merge_and_unload()

        # 3. Save the merged model
        #model.save_pretrained(out_dir_full_model)
        
        print(os.system("nvidia-smi"))
    except Exception as e:
        print(e)
        print(os.system("nvidia-smi"))
if __name__ == "__main__":
    main()
