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
from scripts.formatting_results import *
from scripts.bert_filtering import *
from scripts.negation import *
from scripts.prompting import *
from scripts.utils import *
from scripts.llama_vision_engine import *
#from scripts.llava_med_engine import *
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

def main():
    parser = argparse.ArgumentParser(description="PhenoGPT2 Phenotype Recognizer and Normalizer")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output directory name")
    parser.add_argument("-model_dir", "--model_dir", help="Model directory path")
    parser.add_argument("-lora", "--lora", action="store_true", help="Use LoRA model")
    parser.add_argument("-index", "--index", type=int, help="Index identifier for saving")
    parser.add_argument("-negation", "--negation", action="store_true", help="Allow negation filtering")
    parser.add_argument("-negation_model", "--negation_model", required=False, default='Qwen/Qwen3-4B-Instruct-2507', help="Define the negation model")
    parser.add_argument("--text_only", action="store_true", help="Force using only text module")
    parser.add_argument("--vision_only", action="store_true", help="Force using only vision module")
    parser.add_argument("-vision", "--vision", default="llama-vision", help="Vision model choice (llava-med or llama-vision)")
    parser.add_argument("-wc", "--wc", default=0, type=int, help="Word count per chunk")
    args = parser.parse_args()
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
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = True)
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
    if args.negation:
        negation_tokenizer = AutoTokenizer.from_pretrained(args.negation_model, use_fast = True)
        negation_model = AutoModelForCausalLM.from_pretrained(
            args.negation_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    print('start phenogpt2')
    output_dir = os.path.dirname(model_id) + f"/evaluations/{args.output}/"
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    ## Process Input:
    data_input = read_input(args.input)

    # Load extracted results
    out_path =f"{output_dir}/phenogpt2_rep{args.index}.pkl"
    print(out_path, flush=True)

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
    ###
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
    emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    i = args.index
    seen = 0
    negation = args.negation
    all_responses = {}
    wc = args.wc
    if wc != 0: 
        bert_tokenizer, bert_model = bert_init(local_dir = "./models/bert_filtering/")
    for index, dt in tqdm(data_input.items()):
        all_responses[index] = {}
        if use_text:
            text = data_input[index]['clinical_note'].lower()
            if wc != 0:
                all_chunks = chunking_documents(text, bert_tokenizer, bert_model, word_count = wc)
            else:
                all_chunks = [text]
            temp_response = {}
            for para_id, chunk in enumerate(all_chunks):
                chunk = chunk.replace("'", ""). replace('"', '')
                if len(all_chunks) > 1:
                    pred_label = predict_label(bert_tokenizer, bert_model, {"text":chunk})
                else: # in case users only want to use the whole note for testing
                    pred_label = 'INFORMATIVE'
                if pred_label == 'INFORMATIVE':
                    # Try first attempt
                    response = generate_output(model, tokenizer, chunk, temperature = 0.4, max_new_tokens = 3000, device = device)
                    try:
                        final_response, complete_check = valid_json(response)
                        phenos = final_response.get("phenotypes", {})
                        if not isinstance(phenos, dict) or len(phenos) == 0:
                            raise ValueError("Empty or invalid phenotype dict")
                    except Exception:
                        try:
                            response = generate_output(model, tokenizer, chunk, temperature=0.6, max_new_tokens=5000, device = device)
                            final_response, complete_check = valid_json(response)
                            phenos = final_response.get("phenotypes", {})
                            if not isinstance(phenos, dict) or len(phenos) == 0:
                                raise ValueError("Empty or invalid phenotype dict after retry")
                        except Exception as e:
                            print(f"Error: {e}", flush = True)
                            final_response = {'error_response': response}
                            final_response['pid'] = data_input[index].get('pid', data_input[index].get('pmid', 'unknown'))
                            temp_response[para_id] = final_response
                            continue  # move to the next item
                    if negation:
                        print('Starting detecting negation')
                        try:
                            negation_response = negation_detection(negation_model, negation_tokenizer, chunk, final_response, device = device, max_new_tokens = 10000)
                            final_response = process_negation(final_response, negation_response, complete_check, emb_model)
                        except:
                            final_response['filtered_phenotypes'] = {}
                    else:
                        final_response['filtered_phenotypes'] = {}
                    # if seen <= 10: ## You can comment this out for logging some early results
                    #     if len(final_response['filtered_phenotypes']) > 0:
                    #         print(final_response['filtered_phenotypes'], flush = True)
                    #         print(final_response['negation_analysis'], flush = True)
                    #     else:
                    #         print(final_response['negation_analysis'], flush = True)
                    temp_response[para_id] = final_response
            if len(temp_response) > 1: # if splitting notes into multiple chunks, now merge all
                all_responses[index]['text'] = merge_outputs(temp_response)
            else:
                all_responses[index]['text'] = temp_response[0] # use the whole note as input
        else:
            all_responses[index]['text'] = {}
        seen += 1
        if use_vision:
            vision_phenotypes = phenogpt2_vision.generate_descriptions(dt['image'])
            phen2hpo = generate_output(model, tokenizer, vision_phenotypes, temperature = 0.4, max_new_tokens = 1024, device = device)
            phen2hpo = "{'demographics': {'age': '" + phen2hpo
            phen2hpo = valid_json(phen2hpo)
            phen2hpo = phen2hpo.get("phenotypes", {})
            try:
                phen2hpo = {phen:hpo_dict['HPO_ID'] for phen,hpo_dict in phen2hpo.items()}
            except:
                phen2hpo = {}
            all_responses[index]['image'] = phen2hpo
        else:
            all_responses[index]['image'] = {}
    with open(f'{out_path}', 'wb') as f:
        pickle.dump(all_responses, f)
if __name__ == "__main__":
    main()