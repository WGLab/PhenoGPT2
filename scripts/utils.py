from .prompting import *
import torch
from tqdm import tqdm
import json, glob, re, os, pickle
import numpy as np
from pathlib import Path
#from scripts.PhenotypeMatcher import *
from copy import deepcopy
# Get the directory of the current script (utils.py)
script_dir = Path(__file__).resolve().parent

# Go up one level to the project root and then into the data folder
hpo_db_file = script_dir.parent / 'data' / 'hpo_added_tokens.json'
with open(hpo_db_file, 'r') as f:
    hpo_db = json.load(f)
all_phenotypes = list(hpo_db.keys())
#matcher = PhenotypeMatcher(phenotype_list=all_phenotypes, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
def read_one_file(input_file):
    if isinstance(input_file, list):
        for i,dt in enumerate(input_file):
            if 'input' in input_file[i]:
                input_file[i]['clinical_note'] = input_file[i].pop('input')
            if 'image' not in input_file[i]:
                input_file[i]['image'] = ''
            input_file[i]['pid'] = input_file[i].pop('id')
        input_dict = {dt['pid']:dt for dt in input_file}
        return input_dict
    elif os.path.isfile(input_file):
        if ".pkl" in input_file:
            with open(input_file,'rb') as f:
                input_dict = pickle.load(f)
        elif ".json" in (input_file):
            with open(input_file, 'r') as f:
                input_dict = json.load(f)
        else:
            file_name = input_file.split('/')[-1]#[:-4]
            file_name = file_name.split('.')[0]
            if any([x in input_file for x in ['png', 'jpg', 'jpeg']]):  ## this is image format
                input_text = np.nan
                image_dir = input_file
            else: ## this is text format
                with open(input_file, 'r') as f:
                    input_data = f.readlines()
                    input_data = [d.strip() for d in input_data]
                    input_text = "\t".join(input_data)
                image_dir = np.nan
            input_dict = {file_name:{'clinical_note':input_text, 'image':image_dir, 'pid': file_name}}
        if isinstance(input_dict, list):
            #assert "pid" in input_dict[0], "The given data is missing 'pid' key." 
            if 'clinical_note' in input_dict[0]:
                pass
            elif 'input' in input_dict[0]:
                for i in range(len(input_dict)):
                    input_dict[i]['clinical_note'] = input_dict[i].pop('input')
                assert "clinical_note" in input_dict[0], "The given data is missing 'clinical_note' key."
            #assert "image" in input_dict[0], "The given data is missing 'image' key."
            input_dict = {
                (dt['pid'] if 'pid' in dt else dt['pmid']): dt
                for dt in input_dict
            }

        elif isinstance(input_dict, dict):
            if "pid" in input_dict:
                input_dict_copy = deepcopy(input_dict)
                pid = input_dict_copy['pid']
                input_dict = {pid:{}}
                for k,v in input_dict_copy.items():
                    input_dict[pid][k] = v
            elif "pmid" in input_dict:
                input_dict_copy = deepcopy(input_dict)
                pid = input_dict_copy['pmid']
                input_dict = {pid:{}}
                for k,v in input_dict_copy.items():
                    input_dict[pid][k] = v
            else:
                for k,v in input_dict.items():
                    #assert "pid" in v, "The given data is missing 'pid' key." 
                    assert "clinical_note" in v, "The given data is missing 'clinical_note' key."
                    #assert "image" in v, "The given data is missing 'image' key."
                    break
        else:
            raise ImportError('This file is not in correct format (list or dictionary)')
        return input_dict
    else:
        raise ImportError('This is not a file')
def read_input(input_file):
    if isinstance(input_file, list) or os.path.isfile(input_file):
        return read_one_file(input_file)
    else:
        input_list = glob.glob(input_file + "/*")
    input_dict = {}
    for f in input_list:
        data = read_one_file(f)
        input_dict = {**input_dict, **data}
    return(input_dict)
def generate_output(model, tokenizer, data_point, device, max_new_tokens = 10000, temperature = 0.5, top_p=0.8, negation_detect = False):
    # --------------------------------------------------
    # 1. Build raw prompt string
    # --------------------------------------------------
    if negation_detect:
        messages = prompt_negation(data_point)
        max_new_tokens = min(32768, max_new_tokens * 2)
    else:
        messages = generate_prompt(data_point)
        
    # --------------------------------------------------
    # 2. Apply Qwen chat template (NO thinking)
    # --------------------------------------------------
    config = model.config
    if config.model_type == 'llama':
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # 🚨 critical for Qwen-3
        )
        terminators = [tokenizer.eos_token_id]
        if "<|eot_id|>" in tokenizer.get_vocab():
            terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    blocked_strings = ["HP", "hp", "hP", "Hp"]
    bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in blocked_strings]
    bad_words_ids = [
        ids for ids in bad_words_ids if ids and len(ids) > 0
    ]
    
    # --------------------------------------------------
    # 3. Tokenize once
    # --------------------------------------------------
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # --------------------------------------------------
    # 4. Generate
    # --------------------------------------------------
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=bad_words_ids,
    )

    # --------------------------------------------------
    # 6. Decode ONLY newly generated tokens
    # --------------------------------------------------
    gen_tokens = outputs[0, input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return text
def generate_output_batch(
    model,
    tokenizer,
    data_points,   # list of strings (same as your generate_output input)
    device,
    max_new_tokens=5000,
    temperature=0.5,
    top_p=0.8,
    negation_detect=False
):
    """
    Batched version of generate_output(). Returns list[str] aligned with data_points.
    It preserves:
      - chat template logic (llama vs qwen)
      - terminators / eos handling
      - bad_words_ids
      - decoding only newly generated tokens (per sample)
    """
    if len(data_points) == 0:
        return []

    # --------------------------------------------------
    # 1. Build prompt strings (one per sample)
    # --------------------------------------------------
    prompts = []

    for dp in data_points:
        if negation_detect:
            messages = prompt_negation(dp)
        else:
            messages = generate_prompt(dp)

        config = model.config
        if config.model_type == "llama":
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # critical for Qwen-3
            )
        prompts.append(prompt)

    # --------------------------------------------------
    # 2. Terminators + bad words (same as generate_output)
    # --------------------------------------------------
    config = model.config
    if config.model_type == "llama":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = [tokenizer.eos_token_id]
        if "<|eot_id|>" in tokenizer.get_vocab():
            terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    blocked_strings = ["HP", "hp", "hP", "Hp"]
    bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in blocked_strings]
    bad_words_ids = [ids for ids in bad_words_ids if ids and len(ids) > 0]

    # --------------------------------------------------
    # 3. Tokenize as a batch (padding)
    # --------------------------------------------------
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # True per-sample input lengths (exclude padding)
    # attention_mask is 1 for real tokens, 0 for pad
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    # --------------------------------------------------
    # 4. Generate (one GPU call)
    # --------------------------------------------------
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=bad_words_ids,
        )

    # --------------------------------------------------
    # 5. Decode ONLY newly generated tokens per sample
    # --------------------------------------------------
    texts = []
    for i in range(outputs.shape[0]):
        gen_tokens = outputs[i, input_lens[i]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        texts.append(text)

    return texts

def merge_outputs(data):
    """
    This function is used to combine all results for one note if users decided to split the note into multiple chunks
    """
    pheno_dict = {}
    filtered_pheno_dict = {}
    valid_evidence_demographics_dict = {}
    evidence_demographics_dict = {}
    evidence_phenotypes_dict = {}
    demographics = {}
    check = 0
    has_valid_chunk = False
    last_error = None
    for index, (tid, data_dict) in enumerate(data.items()):
        if 'error_response' not in data_dict:
            has_valid_chunk = True
            if index == check and (len(data_dict) > 0):
                demographics = data_dict['demographics'] # most of demographics are in the first chunk
                try:
                    evidence_demographics_dict = data_dict['negation_analysis']['demographics']
                except:
                    pass
            if 'age' not in data_dict['demographics']:
                data_dict['demographics']['age'] = 'unknown'
            if 'sex' not in data_dict['demographics']:
                data_dict['demographics']['sex'] = 'unknown'
            if 'race' not in data_dict['demographics']:
                data_dict['demographics']['race'] = 'unknown'
            if 'ethnicity' not in data_dict['demographics']:
                data_dict['demographics']['ethnicity'] = 'unknown'
            if len(demographics) > 0 and isinstance(data_dict['demographics']['age'], str) and str(demographics['age']).lower() == 'unknown' and len(data_dict) > 0:
                demographics['age'] = data_dict['demographics']['age']
                try:
                    evidence_demographics_dict['age'] = data_dict['negation_analysis']['demographics']['age']
                except:
                    pass
            if len(demographics) > 0 and isinstance(data_dict['demographics']['sex'], str) and demographics['sex'].lower() == 'unknown' and len(data_dict) > 0:
                demographics['sex'] = data_dict['demographics']['sex']
                try:
                    evidence_demographics_dict['sex'] = data_dict['negation_analysis']['demographics']['sex']
                except:
                    pass
            if len(demographics) > 0 and isinstance(data_dict['demographics']['ethnicity'], str) and demographics['ethnicity'].lower() == 'unknown' and len(data_dict) > 0:
                demographics['ethnicity'] = data_dict['demographics']['ethnicity']
                try:
                    evidence_demographics_dict['ethnicity'] = data_dict['negation_analysis']['demographics']['ethnicity']
                except:
                    pass
            try:
                pheno_dict = {**pheno_dict, **data_dict['phenotypes']}
            except:
                pass
            try:
                filtered_pheno_dict = {**filtered_pheno_dict, **data_dict['filtered_phenotypes']}
            except:
                pass
            try:
                evidence_phenotypes_dict = {**evidence_phenotypes_dict, **data_dict['negation_analysis']['phenotypes']}
            except: ## sometimes the negation analysis cannot be properly formatted/fixed. So, we simply drop them for easier processing.
                pass
            if (len(evidence_demographics_dict) == 0) and ('negation_analysis' in data_dict) and (isinstance(data_dict['negation_analysis'], dict)) and ('demographics' in data_dict['negation_analysis']):
                evidence_demographics_dict = data_dict['negation_analysis']['demographics']
        else:
            last_error = data_dict['error_response']
            
        if len(demographics) == 0:
            demographics = {'age':'unknown', 'sex': 'unknown', 'ethnicity': 'unknown'}
        
    
    negation_dict = {'demographics': evidence_demographics_dict, 'phenotypes': evidence_phenotypes_dict}

    # deduplicating
    pheno_dict_copy = {} 
    check_phens = []
    for k,v in pheno_dict.items():
        if (isinstance(v, dict)) and ('HPO_ID' in v) and (v['HPO_ID'] not in check_phens):
            pheno_dict_copy[k] = v
            check_phens.append(v['HPO_ID'])
    filtered_pheno_dict_copy = {}
    check_phens = []
    for k,v in filtered_pheno_dict.items():
        if (isinstance(v, dict)) and ('HPO_ID' in v) and (v['HPO_ID'] not in check_phens):
            filtered_pheno_dict_copy[k] = v
            check_phens.append(v['HPO_ID'])
    if has_valid_chunk:    
        return {'demographics':demographics,'phenotypes':pheno_dict_copy, 'filtered_phenotypes':filtered_pheno_dict_copy, 'negation_analysis': negation_dict}
    else:
        return {'demographics':demographics,'phenotypes':pheno_dict_copy, 'filtered_phenotypes':filtered_pheno_dict_copy, 'negation_analysis': negation_dict, 'error_response': last_error}
def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

# def impute_missing_hpo(term):
#     match, score = matcher.match(term)
#     hpo_match = hpo_db[match]
#     return hpo_match