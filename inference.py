"""
inference.py
============
PhenoGPT2 — main inference entry point.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scripts.helpers import *

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PhenoGPT2 Phenotype Recognizer and Normalizer")
    parser.add_argument("-i",  "--input",      required=True,  help="Path to input JSON file")
    parser.add_argument("-o",  "--output",     required=True,  help="Output directory name")
    parser.add_argument("-model_dir", "--model_dir",           help="Model directory path")
    parser.add_argument("-lora", "--lora",     action="store_true", help="Use LoRA model")
    parser.add_argument("-index", "--index",   type=int,       help="Index identifier for saving")
    parser.add_argument("-batch_size", "--batch_size",         default=7,  type=int,
                        help="How many samples are processed at the same time")
    parser.add_argument("-chunk_batch_size", "--chunk_batch_size", default=7, type=int,
                        help="Number of chunks processed on GPU at once. Ignored if wc=0")
    parser.add_argument("-negation", "--negation",             action="store_true",
                        help="Allow negation filtering")
    parser.add_argument("-negation_model", "--negation_model", required=False,
                        default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Define the negation model")
    parser.add_argument("-attn_implementation", "--attn_implementation", required=False,
                        default="eager",
                        help="Attention implementation. 'eager' by default.")
    parser.add_argument("--text_only",   action="store_true", help="Force text module only")
    parser.add_argument("--vision_only", action="store_true", help="Force vision module only")
    parser.add_argument("-vision_dir", "--vision_dir", default=None,
                        help="Vision model directory")
    parser.add_argument("-wc", "--wc",         default=0, type=int,
                        help="Word count per chunk")
    args = parser.parse_args()

    # ---- Read input first to determine which modalities are needed ----
    data_input           = read_input(args.input)
    use_text, use_vision = infer_modes(args, data_input)

    # ---- Load only the models needed for active modalities ----
    text_client = build_llm(args) if use_text else None

    # negation + emb_model only needed for text mode
    if use_text:
        negation_client, emb_model = build_negation(args)
    else:
        negation_client = None
        emb_model       = None

    # vision client
    vision_client = build_vision(args, use_vision)

    # emb_model for vision HPO mapping: reuse negation emb_model if available,
    # otherwise load a fresh one (vision-only mode)
    if use_vision and emb_model is None:
        emb_model = build_emb_model_for_vision()

    # ---- BERT chunk filtering (import AFTER all clients are loaded) ----
    wc = args.wc
    from scripts.bert_filtering import bert_init, chunking_documents, predict_label

    if use_text and wc != 0:
        bert_tokenizer, bert_model = bert_init(local_dir="./models/bert_filtering/")
    else:
        bert_tokenizer, bert_model = None, None

    print("start phenogpt2")
    output_dir = args.output
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    out_path = f"{args.output}/phenogpt2_rep{args.index}.pkl"
    print(out_path, flush=True)

    negation      = args.negation
    all_responses = {}

    # ---- DataLoader ----
    allocated_cpus = os.cpu_count() or 1
    num_workers    = max(1, min(allocated_cpus - 1, 4))
    dataset = PhenoGPT2Dataset(data_input)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(allocated_cpus > 1),
        collate_fn=collate_fn,
        prefetch_factor=4 if allocated_cpus > 1 else None,
    )

    seen = 0
    for batch in tqdm(loader, desc="Running Batch"):

        # --- TEXT (batched GPU via vLLM) ---
        if use_text:
            batch_text_results = process_one_batch_text(
                batch=batch,
                data_input=data_input,
                client=text_client,
                bert_tokenizer=bert_tokenizer,
                bert_model=bert_model,
                negation=negation,
                negation_client=negation_client,
                emb_model=emb_model,
                wc=wc,
                chunk_batch_size=args.chunk_batch_size,
            )
        else:
            batch_text_results = {index: {"text": {}, "image": {}} for index, _ in batch}

        # --- VISION ---
        if use_vision:
            vision_indices = []
            vision_paths   = []
            for index, dt in batch:
                if dt.get("image") and pd.notnull(dt["image"]):
                    vision_indices.append(index)
                    vision_paths.append(dt["image"])
                else:
                    batch_text_results[index]["image"] = {}

            if vision_paths:
                # batched vision inference
                vision_phenotypes_list = vision_client.generate_from_paths_batch(vision_paths)

                # embedding-based HPO mapping (no text model needed)
                hpo_mapped_list = map_vision_to_hpo(
                    vision_phenotypes_list, emb_model, threshold=0.8
                )

                for index, phen2hpo in zip(vision_indices, hpo_mapped_list):
                    batch_text_results[index]["image"] = phen2hpo
        else:
            for index, _ in batch:
                batch_text_results[index]["image"] = {}

        # --- Commit results ---
        for index, _ in batch:
            all_responses[index] = batch_text_results[index]
            if seen <= 10:
                print(all_responses[index], flush=True)
            seen += 1

    with open(out_path, "wb") as f:
        pickle.dump(all_responses, f)


if __name__ == "__main__":
    main()