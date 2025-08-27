import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import os, sys, re, torch, json, glob, argparse, gc, ast, pickle, requests
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
def main():
    parser = argparse.ArgumentParser(description="PhenoGPT2 Phenotype Recognizer and Normalizer")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output directory name")
    parser.add_argument("-model_dir", "--model_dir", help="Model directory path")
    parser.add_argument("-lora", "--lora", action="store_true", help="Use LoRA model")
    parser.add_argument("-index", "--index", help="Index identifier for saving")
    parser.add_argument("-negation", "--negation", action="store_true", help="Allow negation filtering")
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
    #tokenizer_id = "Llama3_1/Meta-Llama-3.1-8B-Instruct" # Replace your tokenizer llama directory here
    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast = True)
    model.eval()
    print('start phenogpt2')
    output_dir = os.getcwd() + f"/data/results/{args.output}/"
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    ## Process Input:
    data_input = read_input(args.input)
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
    i = args.index
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
                if len(all_chunks) > 1:
                    pred_label = predict_label(bert_tokenizer, bert_model, {"text":chunk})
                else: # in case users only want to use the whole note for testing
                    pred_label = 'INFORMATIVE'
                if pred_label == 'INFORMATIVE':
                    raw_response = generate_output(model, tokenizer, chunk, temperature = 0.4, max_new_tokens = 5000, device = device)
                    response = "{'demographics': {'age': '" + raw_response
                    # Try first attempt
                    try:
                        final_response = fix_and_parse_json(response)
                        phenos = final_response.get("phenotypes", {})
                        if not isinstance(phenos, dict) or len(phenos) == 0:
                            raise ValueError("Empty or invalid phenotype dict")
                    except Exception:
                        # Retry with alternative prompt
                        try:
                            raw_response = generate_output(model, tokenizer, chunk, temperature=0.4, max_new_tokens=6000, alternative_prompt=True, device = device)
                            response = "{'demographics': {'age': '" + raw_response
                            final_response = fix_and_parse_json(response)
                            phenos = final_response.get("phenotypes", {})
                            if not isinstance(phenos, dict) or len(phenos) == 0:
                                raise ValueError("Empty or invalid phenotype dict after retry")
                        except Exception:
                            final_response = {'error_response': response}
                            final_response['pid'] = data_input[index].get('pid', data_input[index].get('pmid', 'unknown'))
                            temp_response[para_id] = final_response
                            continue  # move to the next item
                    example_removed = ['cleft palate', 'seizures', 'dev delay'] ## here are phenotypes in one-shot in alternative prompt
                    temp_phenotypes = {k:v for k,v in final_response['phenotypes'].items() if (k not in example_removed) or (k in chunk)}
                    final_response['phenotypes'] = temp_phenotypes
                    if negation:
                        phenotypes = list(final_response['phenotypes'].keys())
                        phenotypes = [p.lower() for p in phenotypes]
                        positive_phenotypes = remove_negation(model, tokenizer, chunk, phenotypes, device = device)
                        try:
                            phen_dict = {x:y for x,y in final_response['phenotypes'].items() if x in positive_phenotypes and "HP:" in y['HPO_ID']}
                        except:
                            phen_dict = {x:y for x,y in final_response['phenotypes'].items() if x in positive_phenotypes}
                    else:
                        phen_dict = {}
                    final_response['filtered_phenotypes'] = phen_dict
                    if 'pid' in data_input[index]:
                        final_response['pid'] = data_input[index]['pid']
                    else:
                        final_response['pid'] = data_input[index]['pmid']
                    if 'demographics' in final_response.keys():
                        if ('age' not in final_response['demographics'].keys()) or (final_response['demographics']['age'] == '10-year-old' and '10-year-old' not in chunk):
                            final_response['demographics']['age'] = 'unknown'
                        if 'sex' not in final_response['demographics'].keys():
                            final_response['demographics']['sex'] = 'unknown'
                        if 'ethnicity' not in final_response['demographics'].keys() or (final_response['demographics']['ethnicity'].lower() in ['vietnamese', 'vietnam'] and ('vietnamese' not in chunk or 'vietnam' not in chunk)):
                            final_response['demographics']['ethnicity'] = 'unknown'
                        if 'race' not in final_response['demographics'].keys():
                            final_response['demographics']['race'] = 'unknown'
                    temp_response[para_id] = final_response
            all_responses[index]['text'] = merge_outputs(temp_response)
        else:
            all_responses[index]['text'] = {}
        if use_vision:
            vision_phenotypes = phenogpt2_vision.generate_descriptions(dt['image'])
            phen2hpo = generate_output(model, tokenizer, vision_phenotypes, temperature = 0.4, max_new_tokens = 1024, device = device)
            phen2hpo = "{'demographics': {'age': '" + phen2hpo
            phen2hpo = fix_and_parse_json(phen2hpo)
            phen2hpo = phen2hpo.get("phenotypes", {})
            try:
                phen2hpo = {phen:hpo_dict['HPO_ID'] for phen,hpo_dict in phen2hpo.items()}
            except:
                phen2hpo = {}
            all_responses[index]['image'] = phen2hpo
        else:
            all_responses[index]['image'] = {}
    with open(f'{output_dir}phenogpt2_rep{i}.pkl', 'wb') as f:
        pickle.dump(all_responses, f)
if __name__ == "__main__":
    main()