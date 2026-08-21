import pandas as pd
import os, sys, re, torch, json, glob, argparse, gc, ast, pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
 
from scripts.vllm_engine import VllmClient
from scripts.formatting_results import valid_json
from scripts.negation import negation_detection_batch, process_negation
from scripts.utils import (
    generate_output_batch, merge_outputs, read_input,
    hpo_db, all_phenotypes, embed_and_match_hpo,
)
def shard_dict(merged_output, index, num_shards=30):
    """Split a dictionary into `num_shards` parts using stable modulo sharding."""
    assert 0 <= index < num_shards
    keys = sorted(merged_output.keys())
    shard_keys = [k for i, k in enumerate(keys) if i % num_shards == index]
    return {k: merged_output[k] for k in shard_keys}
 
 
class PhenoGPT2Dataset(Dataset):
    """Minimal dataset wrapper to allow DataLoader prefetching."""
    def __init__(self, data_input):
        self.data_input = data_input
        self.keys = list(data_input.keys())
 
    def __len__(self):
        return len(self.keys)
 
    def __getitem__(self, idx):
        k = self.keys[idx]
        return k, self.data_input[k]
 
 
def collate_fn(batch):
    return batch
 
 
def build_llm(args):
    """Load the main PhenoGPT2 text model as a VllmClient (78% GPU memory)."""
    model_id = args.model_dir if args.model_dir else os.path.join(os.getcwd(), 'models', 'phenogpt2')
 
    if args.lora:
        print(
            "[WARNING] vLLM does not load PEFT LoRA adapters on-the-fly.\n"
            "          Please merge the LoRA weights offline first:\n"
            "          PeftModel.from_pretrained(base, lora_dir).merge_and_unload()\n"
            "          Then point model_dir at the merged checkpoint directory."
        )
 
    return VllmClient(
        model_path=model_id,
        temperature=0.3,
        max_tokens=5000,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=torch.cuda.device_count(),
    )
 
 
def build_negation(args):
    """Load the negation model (15% GPU memory) and embedding model."""
    if args.negation:
        from sentence_transformers import SentenceTransformer
        negation_client = VllmClient(
            model_path=args.negation_model,
            temperature=0.3,
            max_tokens=5000,
            gpu_memory_utilization=0.3,
            max_model_len=50000,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    else:
        negation_client = None
        emb_model = None
    return negation_client, emb_model
 
 
def build_vision(args, use_vision: bool):
    """Load the vision model only when vision mode is active."""
    print(f"use_vision: {use_vision}")
    if use_vision:
        from scripts.vllm_vision_engine import PhenoVisionClient
        return PhenoVisionClient(
            model_path=args.vision_dir,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=torch.cuda.device_count(),
        )
    return None
 
 
def build_emb_model_for_vision():
    """Load embedding model for vision HPO mapping (vision-only mode)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
 
 
def infer_modes(args, data_input):
    """Determine whether to run text and/or vision modules."""
    use_text = use_vision = False
    if args.text_only:
        use_text = True
    elif args.vision_only:
        use_vision = True
    else:
        for dt in data_input.values():
            if pd.notnull(dt.get('clinical_note')): use_text = True
            if pd.notnull(dt.get('image')): use_vision = True
            if use_text and use_vision:
                break
    return use_text, use_vision
 
def process_one_batch_text(
    batch, data_input, client, bert_tokenizer, bert_model,
    negation, negation_client, emb_model, wc, chunk_batch_size=4
):
    """
    Batch text pipeline. Returns dict: {index: {'text': ..., 'image': {}}}
    """
    from scripts.bert_filtering import chunking_documents, predict_label
 
    temp_per_patient = {index: {} for index, _ in batch}
 
    # ---- Build jobs ----
    jobs = []
    for index, dt in batch:
        text = data_input[index]['clinical_note'].lower()
        if wc != 0:
            all_chunks = chunking_documents(text, word_count=wc)
        else:
            chunk_batch_size = len(batch)
            all_chunks = [text]
 
        for para_id, chunk in enumerate(all_chunks):
            chunk = chunk.replace("'", "").replace('"', "")
            pred_label = (
                predict_label(bert_tokenizer, bert_model, {'text': chunk})
                if len(all_chunks) > 1 else 'INFORMATIVE'
            )
            if pred_label == 'INFORMATIVE':
                jobs.append((index, para_id, chunk))
    # ---- First-pass batched generation ----
    outs1 = []
    for i in range(0, len(jobs), chunk_batch_size):
        sub_jobs = jobs[i:i + chunk_batch_size]
        prompts = [chunk for (_, _, chunk) in sub_jobs]
        outs = generate_output_batch(client, prompts, temperature=0.3, max_new_tokens=3000)
        outs1.extend(outs)
    retry_jobs = []
    ok_records = []
 
    for (index, para_id, chunk), response in zip(jobs, outs1):
        try:
            final_response, complete_check = valid_json(response)
            phenos = final_response.get('phenotypes', {})
            if not isinstance(phenos, dict) or len(phenos) == 0:
                raise ValueError('Empty or invalid phenotype dict. Retry!')
            ok_records.append((index, para_id, chunk, final_response, complete_check))
        except Exception:
            retry_jobs.append((index, para_id, chunk)) 
    # ---- Retry batched generation ----
    if retry_jobs:
        outs2 = []
        for i in range(0, len(retry_jobs), chunk_batch_size):
            sub_jobs = retry_jobs[i:i + chunk_batch_size]
            prompts = [chunk for (_, _, chunk) in sub_jobs]
            outs = generate_output_batch(client, prompts, temperature=0.4, max_new_tokens=4000)
            outs2.extend(outs)
 
        for (index, para_id, chunk), response in zip(retry_jobs, outs2):
            try:
                final_response, complete_check = valid_json(response)
                phenos = final_response.get('phenotypes', {})
                if not isinstance(phenos, dict) or len(phenos) == 0:
                    raise ValueError('Empty or invalid phenotype dict after retry.')
                ok_records.append((index, para_id, chunk, final_response, complete_check))
            except Exception:
                final_response = {'error_response': response}
                final_response['pid'] = data_input[index].get('pid', data_input[index].get('pmid', 'unknown'))
                temp_per_patient[index][para_id] = final_response
 
    # ---- Collect negation jobs ----
    neg_chunks, neg_points, neg_keys = [], [], []
 
    for index, para_id, chunk, final_response, complete_check in ok_records:
        if negation and negation_client is not None:
            neg_chunks.append(chunk)
            neg_points.append(final_response)
            neg_keys.append((index, para_id, complete_check))
        else:
            final_response['filtered_phenotypes'] = {}
            temp_per_patient[index][para_id] = final_response
 
    # ---- Batched negation ----
    if negation and neg_points:
        neg_texts = []
        for i in range(0, len(neg_points), chunk_batch_size):
            outs = negation_detection_batch(
                negation_client,
                neg_chunks[i:i + chunk_batch_size],
                neg_points[i:i + chunk_batch_size], max_new_tokens=5000
            )
            neg_texts.extend(outs)
 
        for (index, para_id, complete_check), final_response, neg_text in zip(neg_keys, neg_points, neg_texts):
            try:
                final_response = process_negation(final_response, neg_text, complete_check, emb_model)
            except Exception:
                final_response['filtered_phenotypes'] = {}
                final_response['negation_analysis'] = {}
            temp_per_patient[index][para_id] = final_response
 
    # ---- Merge per patient ----
    batch_results = {}
    for index, _ in batch:
        if len(temp_per_patient[index]) > 1:
            text_out = merge_outputs(temp_per_patient[index])
        else:
            values = list(temp_per_patient[index].values())
            text_out = values[0] if values else {}
        batch_results[index] = {'text': text_out, 'image': {}}
 
    return batch_results