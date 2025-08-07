import os
from datasets import load_dataset, Dataset#, load_from_disk
import datasets
import torch
from tokenizers import AddedToken, pre_tokenizers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from transformers import DataCollatorForSeq2Seq
torch.backends.cuda.matmul.allow_tf32 = True
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm.auto import tqdm
import gc, json, pickle, joblib, argparse
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    PeftModel
)
gc.collect()
torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
device = "cuda" if torch.cuda.is_available() else "cpu"
with open('./data/hpo_added_tokens.json', 'r') as f:
    name2hpo = json.load(f)
def form_json_output(data_point):
    phenotype_dict = {phen:{'HPO_ID':v['HPO_ID'],'onset':v['onset']} for phen,v in data_point['output']['phenotypes'].items() if phen in name2hpo.keys()}
    demographics = data_point['output']['demographics'].copy()
    if demographics['ethnicity'] == 'unknown':
        demographics['race'] = 'unknown'
    return {'demographics':demographics, 'phenotypes':phenotype_dict}

def tokenize(prompt, tokenizer, add_eos_token=True):
    CUTOFF_LEN = 15000 ## Maximum token length for a single input text (roughly 9000 words)
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
def generate_prompt(data_point):
    json_output = form_json_output(data_point)
    instruction = "You are a genetic counselor specializing in extracting demographic details and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. Your task is to provide accurate and concise information without generating random answers. When demographic details or phenotype information is not explicitly mentioned in the input, use 'unknown' as the value."
    question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and the value is a sub-dictionary that contains corresponding HPO identifier (HPO_ID) and the patient’s age (onset) when the phenotype first appeared, if mentioned in the text. If any information is unavailable, return 'unknown' for that field.\nInput: "
    #question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and the value is a list that contains corresponding HPO identifier and followed by the patient’s age (onset) when the phenotype first appeared, if mentioned in the text. If any information is unavailable, return 'unknown' for that field.\nInput: "
    
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    {model_answer}<|eot_id|><|end_of_text|>"""
    
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + data_point['input'],
                                model_answer = "\n|==|Response|==|\n" + str(json_output)
                                )
    return prompt
def generate_and_tokenize_prompt(data_point, tokenizer): ## formulate the input text template and tokenize to numbers
    full_prompt = generate_prompt(data_point) # if just use raw text as input => for pretraining
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    return tokenized_full_prompt
def defining_args():
    return """
    llama 3.1 8B
    any note you want to save here
    """
def main():
    """
    Set training parameters and train model
    """
    parser = argparse.ArgumentParser(description="PhenoGPT2 HPO Aware Pretrain Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-train_data", "--train_data", required = True, help="directory to training dataset")
    parser.add_argument("-eval_data", "--eval_data", required = False, default = None, help="directory to evaluation train dataset")
    parser.add_argument("-name", "--name", required = True, help="directory to output folder")
    parser.add_argument("-lora", "--lora", required = False, action="store_true", help="LoRA finetuning")
    parser.add_argument("-model_dir", "--model_dir", required = False, help="Directory to the Vision Foundation Model")
    args = parser.parse_args()
    if args.model_dir:
        model_name = args.model_dir
    else:
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if args.name:
        out_dir_model = os.getcwd() + '/models/' + args.name
    else:
        out_dir_model = os.getcwd() + '/models/phenogpt2/'
    os.makedirs(out_dir_model, exist_ok=True)
    # with open(out_dir_model + '/params.txt', 'w') as f:
    #     f.write(defining_args())
    print(out_dir_model)
    #model_name = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-8B-Instruct/" # Replace your tokenizer llama directory here
    model=AutoModelForCausalLM.from_pretrained(model_name,do_sample=True, #quantization_config=quantization_config,
                                            attn_implementation="flash_attention_2",
                                            torch_dtype=torch.bfloat16, device_map = 'auto')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    with open(args.train_data, 'rb') as f: # REPLACE YOUR TRAINING DATA HERE
        train_data = pickle.load(f)
    print(generate_prompt(train_data[0]))
    train_data = list(map(lambda x: generate_and_tokenize_prompt(x, tokenizer), tqdm(train_data, desc="Processing train")))
    train_data = {key: [item[key] for item in train_data] for key in train_data[0]}
    train_data = Dataset.from_dict(train_data)
    if args.eval_data:
        do_eval = True
        with open(args.eval_data, 'rb') as f: # REPLACE VALIDATION DATA HERE
            val_data = pickle.load(f)
        val_data = list(map(lambda x: generate_and_tokenize_prompt(x, tokenizer), tqdm(val_data, desc="Processing val")))
        val_data = {key: [item[key] for item in val_data] for key in val_data[0]}
        val_data = Dataset.from_dict(val_data)
    else:
        do_eval = False
    training_args = TrainingArguments(
        output_dir=out_dir_model,
    
        # =================== PERFORMANCE ===================
        per_device_train_batch_size=4,  # H100 can handle this
        gradient_accumulation_steps=20,   # Total batch size: 32 * 4 * 8 = 1024
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
        save_steps=10000,  # More frequent saves, faster recovery
        save_total_limit=2,  # Prevent disk bloating
        
        # =================== EVALUATION ===================
        do_eval=do_eval,
        eval_strategy="steps",
        eval_steps=2000,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=4,
        
        # =================== TRAINING CONTROL ===================
        num_train_epochs=10,  # ~3 epochs likely enough for new-token adaptation
        max_steps=-1,        # Let epochs control the length
        
        warmup_ratio=0.03,  # ~3–5% is enough
        weight_decay=0.05,  # Slightly stronger regularization
        learning_rate=1e-5,
        # =================== DISTRIBUTED ===================
        ddp_find_unused_parameters=False,  # Needed for gradient checkpointing + DDP

        # =================== MISC ===================
        push_to_hub=False,
        resume_from_checkpoint=True  # Auto-resume
    )
    if args.lora:
        LORA_R = 64 #128
        LORA_ALPHA = 128 #256
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
        eval_dataset=val_data,
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
        if (labels >= 128257).sum() > 0:
            print("✅ Found in labels!")
        else:
            print("❌ Missing in labels!")
        break
    trainer.train()
    trainer.save_model(out_dir_model)
    tokenizer.save_pretrained(out_dir_model)
    print(os.system("nvidia-smi"))
if __name__ == "__main__":
    main()
