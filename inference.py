import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import os, sys, re, torch, json, glob, argparse, gc, ast, pickle, requests
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
from tokenizers import AddedToken
from peft import PeftModel, PeftConfig

from torch.utils.data import Dataset, DataLoader

from scripts.formatting_results import *
from scripts.bert_filtering import *
from scripts.negation import *
from scripts.prompting import *
from scripts.utils import *
from scripts.llama_vision_engine import *
# from scripts.llava_med_engine import *


device = "cuda" if torch.cuda.is_available() else "cpu"
gc.collect()
torch.cuda.empty_cache()

def shard_dict(merged_output, index, num_shards=30):
    """
    Split a dictionary into `num_shards` parts using stable modulo sharding.
    If index == num_shards, return the remaining (unassigned) keys.
    """
    assert 0 <= index < num_shards

    keys = sorted(merged_output.keys())  # deterministic order

    shard_keys = [k for i, k in enumerate(keys) if i % num_shards == index]

    return {k: merged_output[k] for k in shard_keys}


class PhenoGPT2Dataset(Dataset):
    """
    Minimal dataset wrapper to allow DataLoader prefetching.
    NOTE: No logic changes—this only changes how items are fed into the loop.
    """
    def __init__(self, data_input):
        self.data_input = data_input
        self.keys = list(data_input.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        return k, self.data_input[k]


def _collate(batch):
    return batch


def build_llm(args):
    ##set up model
    #Model
    if args.model_dir:
        model_id = args.model_dir
    else:
        model_id = os.getcwd() + '/models/phenogpt2'

    if args.lora:
        peft_config = PeftConfig.from_pretrained(model_id)
        # Get path to this file (inference.py)
        current_file = os.path.abspath(__file__)

        # Get path to phenogpt2 root (go up 2 levels: scripts/ -> phenogpt2/)
        project_root = os.path.dirname(current_file)

        # Get path to hpo_aware_pretrain
        hpo_aware_pretrain_dir = os.path.join(project_root, "models", "hpo_aware_pretrain")

        base_model_name = peft_config.base_model_name_or_path if os.path.isfile(peft_config.base_model_name_or_path) else hpo_aware_pretrain_dir
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation= args.attn_implementation
        )
        model = PeftModel.from_pretrained(model, model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation= args.attn_implementation
        )

    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

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

    return model, tokenizer


def build_negation(args):
    if args.negation:
        negation_tokenizer = AutoTokenizer.from_pretrained(args.negation_model, use_fast = True)
        negation_tokenizer.padding_side = "left"
        if negation_tokenizer.pad_token_id is None:
            negation_tokenizer.pad_token = negation_tokenizer.eos_token
        negation_model = AutoModelForCausalLM.from_pretrained(
            args.negation_model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation= args.attn_implementation
        )
        emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    else:
        emb_model = None
        negation_tokenizer = None
        negation_model = None

    return negation_model, negation_tokenizer, emb_model


def infer_modes(args, data_input):
    # Determine processing mode
    use_text = use_vision = False
    if args.text_only:
        use_text = True
    elif args.vision_only:
        use_vision = True
    else:
        # Automatically infer mode based on data
        for dt in data_input.values():
            if pd.notnull(dt.get('clinical_note')): use_text = True
            if pd.notnull(dt.get('image')): use_vision = True
            if use_text and use_vision:
                break  # no need to continue scanning
    return use_text, use_vision


def build_vision(args, use_vision):
    # Vision model setup (only if vision is enabled)
    print(f"use_vision: {use_vision}")
    if use_vision:
        phenogpt2_vision = LLaMA_Generator(os.getcwd() + "/models/llama-vision-phenogpt2")
        # vision_model = args.vision.lower() if args.vision else "llava-med"
        # if vision_model == "llava-med":
        #     phenogpt2_vision = LLaVA_Generator(os.getcwd() + "/models/llava-med-phenogpt2")
        # elif vision_model == "llama-vision":
        #     phenogpt2_vision = LLaMA_Generator(os.getcwd() + "/models/llama-vision-phenogpt2")
        # else:
        #     raise ValueError(f"Unsupported vision model '{vision_model}'. Use 'llava-med' or 'llama-vision'.")
        return phenogpt2_vision
    return None
def process_one_batch_text(
    batch,
    data_input,
    model,
    tokenizer,
    bert_tokenizer,
    bert_model,
    negation,
    negation_model,
    negation_tokenizer,
    emb_model,
    wc,
    chunk_batch_size=4
):
    """
    Batch text pipeline with minimal logic changes:
      - same chunking + BERT filtering
      - same generate_output settings + retry settings
      - same valid_json checks
      - same negation behavior (but batched)
      - same merge_outputs per patient
    Returns dict: {index: {"text": ..., "image": {}}}
    """

    # temp per patient: para_id -> final_response
    temp_per_patient = {index: {} for index, _ in batch}

    # Collect jobs: each is (index, para_id, chunk)
    jobs = []
    for index, dt in batch:
        text = data_input[index]["clinical_note"].lower()
        if wc != 0:
            all_chunks = chunking_documents(text, bert_tokenizer, bert_model, word_count=wc)
        else:
            chunk_batch_size = len(batch) ## each note is considered a chunk itself so use batch size instead
            all_chunks = [text]

        for para_id, chunk in enumerate(all_chunks):
            chunk = chunk.replace("'", "").replace('"', "")

            if len(all_chunks) > 1:
                pred_label = predict_label(bert_tokenizer, bert_model, {"text": chunk})
            else:
                pred_label = "INFORMATIVE"

            if pred_label == "INFORMATIVE":
                jobs.append((index, para_id, chunk))

    # --- First batched generation ---
    prompts1 = [chunk for (_, _, chunk) in jobs]
    outs1 = []
    for i in range(0, len(jobs), chunk_batch_size):
        sub_jobs = jobs[i:i+chunk_batch_size]
        prompts = [chunk for (_,_,chunk) in sub_jobs]
        outs = generate_output_batch(
            model, tokenizer, prompts,
            temperature=0.3,
            max_new_tokens=3000,
            device=device
        )
        outs1.extend(outs)
    retry_jobs = []
    ok_records = []  # (index, para_id, chunk, final_response, complete_check)

    for (index, para_id, chunk), response in zip(jobs, outs1):
        try:
            final_response, complete_check = valid_json(response)
            phenos = final_response.get("phenotypes", {})
            if not isinstance(phenos, dict) or len(phenos) == 0:
                raise ValueError("Empty or invalid phenotype dict. Retry!")
            ok_records.append((index, para_id, chunk, final_response, complete_check))
        except Exception:
            retry_jobs.append((index, para_id, chunk))

    # --- Retry batched generation (only failures) ---
    if len(retry_jobs) > 0:
        prompts2 = [chunk for (_, _, chunk) in retry_jobs]
        outs2 = []
        for i in range(0, len(retry_jobs), chunk_batch_size):
            sub_jobs = retry_jobs[i:i+chunk_batch_size]
            prompts = [chunk for (_,_,chunk) in sub_jobs]
            outs = generate_output_batch(
                model, tokenizer, prompts,
                temperature=0.4,
                max_new_tokens=4000,
                device=device
            )
            outs2.extend(outs)
        for (index, para_id, chunk), response in zip(retry_jobs, outs2):
            try:
                final_response, complete_check = valid_json(response)
                phenos = final_response.get("phenotypes", {})
                if not isinstance(phenos, dict) or len(phenos) == 0:
                    raise ValueError("Empty or invalid phenotype dict after retry. No retry!")
                ok_records.append((index, para_id, chunk, final_response, complete_check))
            except Exception as e:
                final_response = {"error_response": response}
                final_response["pid"] = data_input[index].get("pid", data_input[index].get("pmid", "unknown"))
                temp_per_patient[index][para_id] = final_response

    # Put successful generations into temp_per_patient (but don’t negation yet)
    # Also collect for batched negation
    neg_chunks = []
    neg_points = []
    neg_keys = []  # (index, para_id, complete_check)

    for index, para_id, chunk, final_response, complete_check in ok_records:
        if negation:
            neg_chunks.append(chunk)
            neg_points.append(final_response)
            neg_keys.append((index, para_id, complete_check))
        else:
            final_response["filtered_phenotypes"] = {}
            temp_per_patient[index][para_id] = final_response

    # --- Batched negation ---
    if negation and len(neg_points) > 0:
        neg_texts = []
        for i in range(0, len(neg_points), chunk_batch_size):
            sub_chunks = neg_chunks[i:i+chunk_batch_size]
            sub_points = neg_points[i:i+chunk_batch_size]
            outs = negation_detection_batch(
                negation_model,
                negation_tokenizer,
                sub_chunks,
                sub_points,
                device=device,
                max_new_tokens=5000
            )
            neg_texts.extend(outs)

        for (index, para_id, complete_check), final_response, neg_text in zip(neg_keys, neg_points, neg_texts):
            try:
                negation_response = neg_text
                final_response = process_negation(final_response, negation_response, complete_check, emb_model)
            except:
                final_response["filtered_phenotypes"] = {}
                final_response["negation_analysis"] = {}
            temp_per_patient[index][para_id] = final_response

    # --- Merge per patient exactly as before ---
    batch_results = {}
    for index, _ in batch:
        if len(temp_per_patient[index]) > 1:
            text_out = merge_outputs(temp_per_patient[index])
        else:
            temp_value = list(temp_per_patient[index].values())
            if len(temp_value) > 0:
                text_out = temp_value[0]
            else:
                text_out = {}
        batch_results[index] = {"text": text_out, "image": {}}

    return batch_results


def main():
    parser = argparse.ArgumentParser(description="PhenoGPT2 Phenotype Recognizer and Normalizer")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output directory name")
    parser.add_argument("-model_dir", "--model_dir", help="Model directory path")
    parser.add_argument("-lora", "--lora", action="store_true", help="Use LoRA model")
    parser.add_argument("-index", "--index", type=int, help="Index identifier for saving")
    parser.add_argument("-batch_size", "--batch_size", default=7, type=int, help="How many samples are proccesed at the same time")
    parser.add_argument("-chunk_batch_size", "--chunk_batch_size", default=7, type=int, help="Number of chunks processed on GPU at once. This is ignored if wc=0")
    parser.add_argument("-negation", "--negation", action="store_true", help="Allow negation filtering")
    parser.add_argument("-negation_model", "--negation_model", required=False, default='Qwen/Qwen3-4B-Instruct-2507', help="Define the negation model")
    parser.add_argument("-attn_implementation", "--attn_implementation", required=False, default='eager', help="Default implementation 'eager' is turned on by default. Note that: FlashAttention may not be supported on arm64/aarch64 platforms. Flash Attention helps faster inference and lower memory usage.")
    parser.add_argument("--text_only", action="store_true", help="Force using only text module")
    parser.add_argument("--vision_only", action="store_true", help="Force using only vision module")
    parser.add_argument("-vision", "--vision", default="llama-vision", help="Vision model choice (llava-med or llama-vision)")
    parser.add_argument("-wc", "--wc", default=0, type=int, help="Word count per chunk")
    args = parser.parse_args()

    ## Load PhenoGPT2
    model, tokenizer = build_llm(args)
    ## Load Negation pipeline
    negation_model, negation_tokenizer, emb_model = build_negation(args)

    ## Run BERT model for filtering non-phenotypic chunks
    wc = args.wc
    if wc != 0:
        bert_tokenizer, bert_model = bert_init(local_dir = "./models/bert_filtering/")
    else:
        bert_tokenizer, bert_model = None, None

    print('start phenogpt2')
    output_dir = args.output

    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    ## Process Input:
    data_input = read_input(args.input)

    # Load extracted results
    out_path = f"{args.output}/phenogpt2_rep{args.index}.pkl"
    print(out_path, flush=True)

    use_text, use_vision = infer_modes(args, data_input)

    phenogpt2_vision = build_vision(args, use_vision)

    i = args.index
    negation = args.negation
    all_responses = {}

    # ----------------------------
    # DataLoader wrapper (prefetch)
    # ----------------------------
    allocated_cpus = os.cpu_count() or 1
    num_workers = max(1, min(allocated_cpus - 1, 4))
    dataset = PhenoGPT2Dataset(data_input)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if (os.cpu_count() or 0) > 1 else False,
        collate_fn=_collate,
        prefetch_factor=4 if (os.cpu_count() or 0) > 1 else None,
    )

    seen=0
    for batch in tqdm(loader, desc='Running Batch'):
        # --- TEXT (batched GPU) ---
        if use_text:
            batch_text_results = process_one_batch_text(
                batch=batch,
                data_input=data_input,
                model=model,
                tokenizer=tokenizer,
                bert_tokenizer=bert_tokenizer,
                bert_model=bert_model,
                negation=negation,
                negation_model=negation_model,
                negation_tokenizer=negation_tokenizer,
                emb_model=emb_model,
                wc=wc,
                chunk_batch_size=args.chunk_batch_size
            )
        else:
            batch_text_results = {index: {"text": {}, "image": {}} for index, _ in batch}

        # --- VISION ---
        if use_vision:
            for index, dt in batch:
                vision_phenotypes = phenogpt2_vision.generate_descriptions(dt['image'])
                phen2hpo = generate_output(model, tokenizer, vision_phenotypes, temperature=0.4, max_new_tokens=1024, device=device)
                phen2hpo = "{'demographics': {'age': '" + phen2hpo
                phen2hpo = valid_json(phen2hpo)
                phen2hpo = phen2hpo.get("phenotypes", {})
                try:
                    phen2hpo = {phen: hpo_dict['HPO_ID'] for phen, hpo_dict in phen2hpo.items()}
                except:
                    phen2hpo = {}
                batch_text_results[index]["image"] = phen2hpo
        else:
            for index, dt in batch:
                batch_text_results[index]["image"] = {}

        # --- commit results per patient index ---
        for index, _ in batch:
            all_responses[index] = batch_text_results[index]

            if seen <= 10:
                print(all_responses[index], flush=True)
            seen += 1

    with open(f'{out_path}', 'wb') as f:
        pickle.dump(all_responses, f)


if __name__ == "__main__":
    main()
