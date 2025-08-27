from scripts.prompting import *
import torch
from tqdm import tqdm
import json, glob
from pathlib import Path
from scripts.PhenotypeMatcher import *
from copy import deepcopy
# Get the directory of the current script (utils.py)
script_dir = Path(__file__).resolve().parent

# Go up one level to the project root and then into the data folder
hpo_db_file = script_dir.parent / 'data' / 'hpo_added_tokens.json'
with open(hpo_db_file, 'r') as f:
    hpo_db = json.load(f)
all_phenotypes = list(hpo_db.keys())
matcher = PhenotypeMatcher(phenotype_list=all_phenotypes, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
def read_one_file(input_file):
    if os.path.isfile(input_file):
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
            assert "pid" in input_dict[0], "The given data is missing 'pid' key." 
            assert "clinical_note" in input_dict[0], "The given data is missing 'clinical_note' key."
            assert "image" in input_dict[0], "The given data is missing 'image' key."
            input_dict = {dt['pid']:dt for dt in input_dict}
        elif isinstance(input_dict, dict):
            if "pid" in input_dict:
                input_dict_copy = deepcopy(input_dict)
                pid = input_dict_copy['pid']
                input_dict = {pid:{}}
                for k,v in input_dict_copy.items():
                    input_dict[pid][k] = v
            else:
                for k,v in input_dict.items():
                    assert "pid" in v, "The given data is missing 'pid' key." 
                    assert "clinical_note" in v, "The given data is missing 'clinical_note' key."
                    assert "image" in v, "The given data is missing 'image' key."
                    break
        else:
            raise ImportError('This file is not in correct format (list or dictionary)')
        return input_dict
    else:
        raise ImportError('This is not a file')
def read_input(input_file):
    if os.path.isfile(input_file):
        return read_one_file(input_file)
    else:
        input_list = glob.glob(input_file + "/*")
    input_dict = {}
    for f in input_list:
        file_name = f.split('/')[-1]#[:-4]
        file_name = file_name.split('.')[0]
        data = read_one_file(input_file)
        input_dict[file_name] = data
    return(input_dict)
def generate_output(model, tokenizer, data_point, device, max_new_tokens = 10000, temperature = 0.5, top_p=0.8, negation_detect = False, alternative_prompt = False):
    if negation_detect:
        prompt = prompt_negation(data_point)
    else:
        if alternative_prompt:
            prompt = generate_prompt_alternative(data_point)
        else:
            prompt = generate_prompt(data_point)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    #model.to(device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    # Tokens to inhibit
    blocked_strings = ["HP", "hp"]
    bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in blocked_strings]

    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True if temperature > 0 else False,
            temperature=temperature, #higher temperature => generate more creative answers but the responses may change
            top_p=0.8, #higher top_p => less likely to generate random answer not in the text
            bad_words_ids=bad_words_ids,
    )

    response = generation_output[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True).strip()
    if len(input_ids[0]) > 128000:
        print("WARNING: Your text input has more than the predefined maximum 128000 tokens. The results may be defective.")
    return(output)
def merge_outputs(data):
    """
    This function is used to combine all results for one note if users decided to split the note into multiple chunks
    """
    pheno_dict = {}
    filtered_pheno_dict = {}
    demographics = {}
    check = 0
    for index, (tid, data_dict) in tqdm(enumerate(data.items()), desc = "Merging all chunks together"):
        if 'pid' in data_dict:
            pid = data_dict['pid']
        elif 'pmid' in data_dict:
            pid = data_dict['pmid']
        else:
            pid = 'unknown'
        if 'error_response' in data_dict.keys():
            check += 1
            if len(demographics) == 0:
                demographics = {'age':'unknown', 'sex': 'unknown', 'ethnicity': 'unknown', 'race': 'unknown'}
            continue
        if index == check and len(data_dict) > 0:
            demographics = data_dict['demographics'] # most of demographics are in the first chunk
        if 'age' not in data_dict['demographics']:
            data_dict['demographics']['age'] = 'unknown'
        if 'sex' not in data_dict['demographics']:
            data_dict['demographics']['sex'] = 'unknown'
        if 'race' not in data_dict['demographics']:
            data_dict['demographics']['race'] = 'unknown'
        if 'ethnicity' not in data_dict['demographics']:
            data_dict['demographics']['ethnicity'] = 'unknown'
        if len(demographics) > 0 and isinstance(data_dict['demographics']['age'], str) and demographics['age'].lower() == 'unknown' and len(data_dict) > 0:
            demographics['age'] = data_dict['demographics']['age']
        if len(demographics) > 0 and isinstance(data_dict['demographics']['sex'], str) and demographics['sex'].lower() == 'unknown' and len(data_dict) > 0:
            demographics['sex'] = data_dict['demographics']['sex']
        if len(demographics) > 0 and isinstance(data_dict['demographics']['race'], str) and demographics['race'].lower() == 'unknown' and len(data_dict) > 0:
            demographics['race'] = data_dict['demographics']['race']
        if len(demographics) > 0 and isinstance(data_dict['demographics']['ethnicity'], str) and demographics['ethnicity'].lower() == 'unknown' and len(data_dict) > 0:
            demographics['ethnicity'] = data_dict['demographics']['ethnicity']
        try:
            pheno_dict = {**pheno_dict, **data_dict['phenotypes']}
        except:
            pass
        try:
            filtered_pheno_dict = {**filtered_pheno_dict, **data_dict['filtered_phenotypes']}
        except:
            pass
        
        if len(demographics) == 0:
            demographics = {'age':'unknown', 'sex': 'unknown', 'ethnicity': 'unknown', 'race': 'unknown'}

    return {'demographics':demographics,'phenotypes':pheno_dict, 'filtered_phenotypes':filtered_pheno_dict, 'pid':pid}

def impute_missing_hpo(term):
    match, score = matcher.match(term)
    hpo_match = hpo_db[match]
    return hpo_match