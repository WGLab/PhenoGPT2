import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os, sys, re, torch, json, glob, argparse, gc, ast, pickle
from itertools import chain
from datasets import load_dataset
from tokenizers import AddedToken, pre_tokenizers
import numpy as np
from tqdm.auto import tqdm
from ast import literal_eval
import spacy
import medspacy, requests
from medspacy.context import ConText, ConTextRule
from medspacy.ner import TargetRule
from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer, util
import os, re
import warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# Load an empty spaCy model (we don't need entity recognition)
nlp = spacy.blank("en")
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_model = SentenceTransformer('BAAI/bge-en-icl', device=device)

parser = argparse.ArgumentParser(description="PhenoGPT2 Phenotypic Term Detector",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-model_dir", "--model_dir", required = True, help="directory to model folder")
parser.add_argument("-name", "--name", required = True, help="Name of this dataset")
parser.add_argument("-negation", "--negation", action="store_true", required = False, help="Negation results will be considered instead")
args = parser.parse_args()

with open('/home/nguyenqm/projects/CHOPNLP/hp.json', 'r') as f:
    hpo_database = json.load(f)
parent_terms= requests.get("https://ontology.jax.org/api/hp/terms/HP:0000001/children").json()
parent_terms = [t['id'] for t in parent_terms]
hpo_rela = pd.DataFrame(hpo_database['graphs'][0]['edges'])
hpo_rela['sub'] = hpo_rela['sub'].apply(lambda x: x.split("/")[-1].replace("_",":"))
hpo_rela['obj'] = hpo_rela['obj'].apply(lambda x: x.split("/")[-1].replace("_",":"))
hpo_rela = hpo_rela.groupby('sub')['obj'].apply(list).to_dict()
with open('/home/nguyenqm/projects/github/PhenoGPT2/hpo_added_tokens.json','r') as f:
    name2hpo_dict = json.load(f)
name2hpo_dict = {k.lower():v for k,v in name2hpo_dict.items()}
def get_name(hpo_id, name2hpo_dict = name2hpo_dict):
    if hpo_id in name2hpo_dict.keys():
        return name2hpo_dict[hpo_id].lower()
    try:
        return requests.get(f"https://ontology.jax.org/api/hp/terms/{hpo_id}").json()['name'].lower()
    except:
        return ''
def find_paths(node, path=[], mtype = 'actual'):
    # Add current node to the path
    try:
        path.append(node)
        # Check if the current node is the root node
        if node == "HP:0000001":
            #print(" -> ".join(reversed(path)))  # Print the path in reverse order
            #print(path)
            return [path]
        # Recursively explore each parent node
        path_list = []
        if node in list(hpo_rela.keys()):
            parents_id = hpo_rela[node]
        else:
            parents = requests.get(f"https://ontology.jax.org/api/hp/terms/{node}/parents").json()
            parents_id = [x['id'] for x in parents]
        for parent in parents_id:
            temp_paths = find_paths(parent, path.copy(), mtype = 'actual')
            for new_path in temp_paths:
                path_list.append(new_path)
#         parent = parents_id[0]
#         temp_paths = find_paths(parent, path.copy())
#         path_list.append(temp_paths)
    except:
        print("error here:")
        print(mtype)
        print(node)
        print()
        pass
    return(path_list)
def calc_matching_score(actual_paths, pred_paths, metrics_type = 'tree'):
    score_list = []
    for actual_p in actual_paths:
        for pred_p in pred_paths:
            x = np.in1d(actual_p,pred_p)
            true_indices = np.argwhere(x).flatten()
            first_true_index = true_indices[0] # as every phenotype always has the root as the common ancestors
            if metrics_type == 'tree':
                if len(actual_p) >= len(pred_p):
                    score = (len(actual_p)-first_true_index)/len(actual_p)
                else:
                    score = (len(actual_p)-first_true_index)/len(pred_p)
            else:
                actual_dist_diff = first_true_index
                pred_dist_diff = np.argwhere(np.array(pred_p) == actual_p[first_true_index])[0][0]
                score = actual_dist_diff + pred_dist_diff
            score_list.append(score)
    if len(score_list) > 0:
        if metrics_type == 'tree':
            return np.max(score_list)
        else:
            return np.min(score_list)
    else:
        if metrics_type == 'tree':
            return 0
        else:
            return 10

def similarity_score(entity, selected_nodes, similarity_threshold = 0.9, device = device):
    nodes_embeddings = emb_model.encode(selected_nodes, convert_to_tensor=True).to(device)

    # Compute embeddings for the entity and graph_nodes
    entity_embedding = emb_model.encode(entity, convert_to_tensor=True).to(device)

    # Compute cosine similarities
    similarities = util.cos_sim(entity_embedding, nodes_embeddings).squeeze(0)
    similar_terms = [
            node for node, sim in zip(selected_nodes, similarities) if sim >= similarity_threshold
        ]
    similar_terms.sort(key=lambda x: x[1], reverse=True)
    return similar_terms
def similarity_by_hpoid(entity, dict1, selected_nodes, dict2):
    # dict1 contains entity
    # dict2 contains selected_nodes
    for k,v in dict2.items():
        if k in selected_nodes and dict1[entity] == v:
            return k
        #elif k in selected_nodes and calculate_score(...):
            #return k
    return selected_nodes[0]
    
def filter_by_hpo_id(remaining_dict1, remaining_dict2):
    hpo_overlap_dict = {}
    #remaining_dict1 = {k:dict1[k] for k in remaining_dict1.keys()}
    #remaining_dict2 = {k:dict2[k] for k in remaining_dict2.keys()}
    final_remaining_dict1 = {}
    final_remaining_dict2 = {}

    matched_keys2 = set()

    for key1, hpo1 in remaining_dict1.items():
        found_overlap = False
        for key2, hpo2 in remaining_dict2.items():
            if hpo1 == hpo2:
                hpo_overlap_dict[key1] = key2
                matched_keys2.add(key2)
                found_overlap = True
                break
        
        if not found_overlap:
            final_remaining_dict1[key1] = hpo1

    for key2, hpo2 in remaining_dict2.items():
        if key2 not in matched_keys2:
            final_remaining_dict2[key2] = hpo2
    
    return hpo_overlap_dict, final_remaining_dict1, final_remaining_dict2
def match_by_close_hpoid(remaining_dict1, remaining_dict2):
    final_remaining_dict1 = {}
    final_remaining_dict2 = {}
    hpo_overlap_dict = {}
    matched_keys2 = set()
    for key1, hpo1 in remaining_dict1.items():
        best_score = 0
        for key2, hpo2 in remaining_dict2.items():
            actual_paths = find_paths(hpo1, path=[], mtype = 'actual')
            pred_paths = find_paths(hpo2, path=[], mtype = 'phenogpt2')
            tree_score = calc_matching_score(actual_paths, pred_paths, metrics_type = 'tree')
            if tree_score > 0.7 and tree_score > best_score:
                hpo_overlap_dict[key1] = key2
                matched_keys2.add(key2)
                best_score = tree_score
        if best_score == 0:        
            final_remaining_dict1[key1] = hpo1

    for key2, hpo2 in remaining_dict2.items():
        if key2 not in matched_keys2:
            final_remaining_dict2[key2] = hpo2
    return hpo_overlap_dict, final_remaining_dict1, final_remaining_dict2
    
def separate_dicts(dict1, dict2, threshold = 0.9):
    overlap_dict = {}
    remaining_dict1 = {}
    remaining_dict2 = {}

    matched_keys2 = set()  # Track keys in dict2 that matched
    
    for key1,val1 in dict1.items():
        if key1 not in matched_keys2:
            if key1 in dict2.keys():
                overlap_dict[key1] = key1
                matched_keys2.add(key1)
            else:
                selected_nodes = list(set(list(dict2.keys())) - matched_keys2)
                if len(selected_nodes) == 0:
                    remaining_dict1[key1] = val1
                else:
                    similar_nodes = similarity_score(key1, selected_nodes, threshold)
                    if len(similar_nodes) == 0:
                        remaining_dict1[key1] = val1
                    elif len(similar_nodes) == 1:
                        overlap_dict[key1] = similar_nodes[0]
                        matched_keys2.add(similar_nodes[0])
                        #matched_keys2.add(key1)
                    else:
                        best_match = similarity_by_hpoid(key1, dict1, similar_nodes, dict2)
                        overlap_dict[key1] = similar_nodes[0]
                        matched_keys2.add(similar_nodes[0])               

    # Add remaining keys from dict2 to remaining_dict2
    for key2 in dict2:
        if key2 not in matched_keys2:
            remaining_dict2[key2] = dict2[key2]
    
    # Do one more filter to check if there are any phenotypes having the same HPO IDs but slightly different in text
    hpo_overlap_dict, final_remaining_dict1, final_remaining_dict2 = filter_by_hpo_id(remaining_dict1, remaining_dict2)
    hpo_overlap_dict2, final_remaining_dict1, final_remaining_dict2 = match_by_close_hpoid(final_remaining_dict1, final_remaining_dict2)
    
    combined_dict = {**overlap_dict, **hpo_overlap_dict, **hpo_overlap_dict2}
    return combined_dict, final_remaining_dict1, final_remaining_dict2
def merge_dictionaries(dicts, code_check=None):
    #dicts = [dict1, dict2, dict3]
    # Step 1: Identify unique keys and store HPO IDs for all variants
    combined_keys = defaultdict(list)
    dicts = [d for d in dicts if len(d) > 0]
    if len(dicts) == 0:
        return {}
    for d in dicts:
        for k, v in d.items():
            # Check if the key already has a match in combined_keys
            matched_key = None
            for existing_key in combined_keys:
                if k in existing_key or existing_key in k:
                    # Found a duplicate key (exact or partial)
                    matched_key = existing_key if len(existing_key) >= len(k) else k
                    break
            
            # If no match found, use the key as is
            matched_key = matched_key or k
            combined_keys[matched_key].append(v)
    final_dict = {}
    for phen_text, hpoids in combined_keys.items():
        hpoids = [h for h in hpoids if isinstance(h ,str) and h.startswith("HP:")]
        hpoids = list(set(hpoids))
        if len(hpoids) == 0:
            continue
        elif len(hpoids) == 1:
            final_dict[phen_text] = hpoids[0]
        else:
            dbname2hpo = {get_name(hpo_id):hpo_id for hpo_id in hpoids}
            dbname2hpo = {x:y for x,y in dbname2hpo.items() if len(x) > 0}
            if len(dbname2hpo) > 0:
                similar_terms = similarity_score(phen_text, list(dbname2hpo.keys()), 0)
                final_dict[phen_text] = dbname2hpo[similar_terms[0]]
    return final_dict
code_check = list(hpo_rela.keys())
def get_results(true_ann_dict, pred_ann_dict):
    total_tree_score = 0
    total_node_score = 0
    overlap, remaining1, remaining2 = separate_dicts(true_ann_dict, pred_ann_dict)
    check_pred = [] # some true labels are essentially the same but recorded multiple times => but PhenoGPT2 only records once => causing higher precision
    pred_length = len(pred_ann_dict)
    if len(overlap) > 0:
        for actual,pred in overlap.items():
            #try:
            if pred in check_pred:
                pred_length += 1
            else:
                check_pred.append(pred)
            actual_code = true_ann_dict[actual]
            pred_code = pred_ann_dict[pred]
            actual_paths = find_paths(actual_code, path=[], mtype = 'actual')
            pred_paths = find_paths(pred_code, path=[], mtype = 'phenogpt2')
            if len(pred_paths) > 0:
                tree_score = calc_matching_score(actual_paths, pred_paths, metrics_type = 'tree')
#                 print(actual)
#                 print(tree_score)
                total_tree_score += tree_score
                node_score = calc_matching_score(actual_paths, pred_paths, metrics_type = 'node')
                total_node_score += node_score
            else:
                total_tree_score += 0
                total_node_score += 10
        recall = round(len(overlap)/len(true_ann_dict),2)
        precision = round(len(overlap)/pred_length,2)
        if (recall+precision) == 0:
            f1score = 0
        else:
            f1score = round(2*recall*precision/(recall+precision),2)
        conversion_rate_tree = round(total_tree_score/len(overlap),2)
        conversion_rate_node = round(total_node_score/len(overlap),2)
        return recall, precision, f1score, conversion_rate_tree, conversion_rate_node, len(overlap)
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
def analyze_results(dir0, dir1, dir2, input_data, negation = False, merging_only = False):
    try:
        if ".json" in dir0:
            with open(dir0, 'r') as f:
                all_responses0 = json.load(f)
        else:
            with open(dir0, 'rb') as f:
                all_responses0 = pickle.load(f)
        all_responses0 = {str(k):v for k,v in all_responses0.items()}
    except:
        all_responses0 = {}
        
    try:
        if ".json" in dir1:
            with open(dir1, 'r') as f:
                all_responses1 = json.load(f)
        else:
            with open(dir1, 'rb') as f:
                all_responses1 = pickle.load(f)
        all_responses1 = {str(k):v for k,v in all_responses1.items()}
    except:
        all_responses1 = {}        
    try:
        if ".json" in dir2:
            with open(dir2, 'r') as f:
                all_responses2 = json.load(f)
        else:
            with open(dir2, 'rb') as f:
                all_responses2 = pickle.load(f)
        all_responses2 = {str(k):v for k,v in all_responses2.items()}
    except:
        all_responses2 = {}
    result_metrics = {}
    all_phenotypes_results = {}
    if negation:
        phen_field = 'filtered_phenotypes'
    else:
        phen_field = 'phenotypes'
    for index, result in tqdm(input_data.items()):
        true_ann_dict = input_data[str(index)]['phenotypes']
        if isinstance(true_ann_dict, str):
            true_ann_dict = literal_eval(true_ann_dict)
        all_phenotype_dicts = []
        if len(all_responses0) > 0:
            #if phen_field in all_responses0[str(index)].keys() :
            if type(list(all_responses0.keys())[0]) == str:
                index = str(index)
            else:
                index = int(index)
            if phen_field in all_responses0[index].keys() :
                rep0 = dict(all_responses0[index][phen_field])
                if type(rep0) == dict:
                    rep0 = {k:v['HPO_ID'] if (isinstance(v, dict) and "HPO_ID" in v.keys()) else v for k,v in rep0.items() }
                    all_phenotype_dicts.append(rep0)
        if len(all_responses1) > 0:
            if type(list(all_responses1.keys())[0]) == str:
                index = str(index)
            else:
                index = int(index)
            if phen_field in all_responses1[index].keys():
                rep1 = dict(all_responses1[index][phen_field])
                if type(rep1) == dict:
                    rep1 = {k:v['HPO_ID'] if (isinstance(v, dict) and "HPO_ID" in v.keys()) else v for k,v in rep1.items()}
                    all_phenotype_dicts.append(rep1)
        if len(all_responses2) > 0:
            if type(list(all_responses2.keys())[0]) == str:
                index = str(index)
            else:
                index = int(index)
            if phen_field in all_responses2[index].keys():
                rep2 = dict(all_responses2[index][phen_field])
                if type(rep2) == dict:
                    rep2 = {k:v['HPO_ID'] if (isinstance(v, dict) and "HPO_ID" in v.keys()) else v for k,v in rep2.items()}
                    all_phenotype_dicts.append(rep2)
            
        pred_ann_dict = merge_dictionaries(all_phenotype_dicts, code_check)
        all_phenotypes_results[index] = pred_ann_dict
        if merging_only:
            continue
        empty_ids = []
        if len(pred_ann_dict) == 0: # no generation
            result_metrics[index] = {}
            empty_ids.append(index)
            continue
        recall, precision, f1score, conversion_rate_tree, conversion_rate_node, n_overlap = get_results(true_ann_dict, pred_ann_dict)
        if not all(pd.isna([recall, precision, f1score, conversion_rate_tree, conversion_rate_node, n_overlap])):
            if recall >= 0.9 and conversion_rate_tree > 0.95 and n_overlap >= 7:
                example = True
            else:
                example = False  
            result_metrics[index] = {'recall':recall, 'precision':precision, 'f1':f1score, 'tree':conversion_rate_tree, 'node':conversion_rate_node, 'example':example}
        else:
            result_metrics[index] = {}
    if merging_only:
        return all_phenotypes_results, {}, [] 

    return all_phenotypes_results, result_metrics, empty_ids
def process_true_ann(true_ann, sep = ' | '):
    if len(true_ann) > 0:
        true_ann_dict = [t.strip().split(sep) for t in true_ann]
        return {t[0]:t[1] for t in true_ann_dict}
    else:
        return {}
def print_results(result_metrics, divide_total = False):
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_tree = 0
    total_node = 0
    empty_files = []
    for k,metrics in result_metrics.items():
        if len(metrics) > 0:
            total_recall += metrics['recall']
            total_precision += metrics['precision']
            total_f1 += metrics['f1']
            total_tree += metrics['tree']
            total_node += metrics['node']
        else:
            empty_files.append(k)
    if divide_total:
        n = 0
    else:
        n = len(empty_files)
    print(f"Total number of samples: {len(result_metrics)}")
    print(f"Total number of Defined samples: {len(result_metrics) - len(empty_files)}")
    print(f"Undetermined Samples: {empty_files}")
    print(f"% Recall: {total_recall/(len(result_metrics)-n)}")
    print(f"% Precision: {total_precision/(len(result_metrics)-n)}")
    print(f"% F1-Score: {total_f1/(len(result_metrics)-n)}")
    print(f"Average Tree Score: {total_tree/(len(result_metrics)-n)}")
    print(f"Average Node Score: {total_node/(len(result_metrics)-n)}")
def process_phenotagger_results(phenotagger_dir, negation = False):
    with open(phenotagger_dir, 'r') as f:
        texts = f.readlines()
    texts = [t.strip() for t in texts if len(t.strip()) > 0]
    if len(texts) > 0:
        df = pd.DataFrame([x.strip().split('\t')[3:] for x in texts[2:]], columns = ['phenotype','category','hpo','score','filter'])
        if negation:
            df = df.loc[df['filter'] == 'positive', ['phenotype','hpo']]
        else:
            df = df[['phenotype','hpo']]
        pred_ann_dict = df.set_index('phenotype')['hpo'].to_dict()
    else:
        pred_ann_dict = {}
    return pred_ann_dict
def analyze_phenotagger_results(phenotagger_dirs, input_data, negation):
    result_metrics = {}
    all_phenotypes_results = {}
    for phenotagger_dir in tqdm(phenotagger_dirs):
        file_id = phenotagger_dir.split('/')[-1].split(".")[0]
        true_ann_dict = input_data[file_id]['phenotypes']
        if isinstance(true_ann_dict, str):
            true_ann_dict = literal_eval(true_ann_dict)
        pred_ann_dict = process_phenotagger_results(phenotagger_dir, negation)
        all_phenotypes_results[file_id] = pred_ann_dict
        empty_ids = []
        if len(pred_ann_dict) == 0: # no generation
            result_metrics[file_id] = {}
            empty_ids.append(file_id)
            continue
        recall, precision, f1score, conversion_rate_tree, conversion_rate_node, n_overlap = get_results(true_ann_dict, pred_ann_dict)
        if not all(pd.isna([recall, precision, f1score, conversion_rate_tree, conversion_rate_node, n_overlap])):
            if recall >= 0.9 and conversion_rate_tree > 0.95 and n_overlap >= 7:
                example = True
            else:
                example = False  
            result_metrics[file_id] = {'recall':recall, 'precision':precision, 'f1':f1score, 'tree':conversion_rate_tree, 'node':conversion_rate_node, 'example':example}
        else:
            result_metrics[file_id] = {}
    return all_phenotypes_results, result_metrics, empty_ids
def analyze_phenogpt_results(phenogpt_dict, input_data):
    result_metrics = {}
    all_phenotypes_results = {}
    for file_id,pred_ann_dict in tqdm(phenogpt_dict.items()):
        true_ann_dict = input_data[file_id]['phenotypes']
        if isinstance(true_ann_dict, str):
            true_ann_dict = literal_eval(true_ann_dict)
        all_phenotypes_results[file_id] = pred_ann_dict
        empty_ids = []
        if len(pred_ann_dict) == 0: # no generation
            result_metrics[file_id] = {}
            empty_ids.append(file_id)
            continue
        recall, precision, f1score, conversion_rate_tree, conversion_rate_node, n_overlap = get_results(true_ann_dict, pred_ann_dict)
        if not all(pd.isna([recall, precision, f1score, conversion_rate_tree, conversion_rate_node, n_overlap])):
            if recall >= 0.9 and conversion_rate_tree > 0.95 and n_overlap >= 7:
                example = True
            else:
                example = False  
            result_metrics[file_id] = {'recall':recall, 'precision':precision, 'f1':f1score, 'tree':conversion_rate_tree, 'node':conversion_rate_node, 'example':example}
        else:
            result_metrics[file_id] = {}
    return all_phenotypes_results, result_metrics, empty_ids

def find_matching_file(directory, pattern="phenogpt2_rep\\d+"):
    for f in os.listdir(directory):
        if f.endswith(('.pkl', '.json')) and re.search(pattern, f):
            return os.path.join(directory, f)
    return None

def process_results(model_dir, dataset, true_ann, negation = False):
    dir0 = find_matching_file(model_dir + f'{dataset}/', 'phenogpt2_rep0')
    dir1 = find_matching_file(model_dir + f'{dataset}/', 'phenogpt2_rep1')
    dir2 = find_matching_file(model_dir + f'{dataset}/', 'phenogpt2_rep2')
    negation_phenotypes, negation_results, empty_ids = analyze_results(dir0, dir1, dir2, true_ann, negation)
    return negation_phenotypes, negation_results, empty_ids
def save_data(model_dir, dataset, negation_phenotypes, negation_results, empty_ids, negation = False):
    saved_dir = model_dir + f'evaluation/{dataset}/'
    os.makedirs(saved_dir, exist_ok = True)
    if negation:
        postfix = '_negated'
    else:
        postfix = ''
    with open(saved_dir+f'processed_phenotypes{postfix}.json', 'w') as f:
        json.dump(negation_phenotypes, f)
    with open(saved_dir+f'processed_results{postfix}.json', 'w') as f:
        json.dump(negation_results, f)
    if empty_ids:
        with open(saved_dir+f'empty_ids{postfix}.json', 'w') as f:
            json.dump(empty_ids, f)

def main():
    model_dir = args.model_dir
    name = args.name
    negation = args.negation
    if 'arcus' in name.lower():
        with open('/home/nguyenqm/projects/MedicalDBs/Arcus/phenogpt2/arcus_notes_phenogpt2_updated.json', 'r') as f:
            arcus_ann = json.load(f)
        input_data = {}
        for i, ann in enumerate(arcus_ann):
            input_data[str(i)] = ann
    elif 'phenopackage' in name.lower():
        with open('/home/nguyenqm/projects/MedicalDBs/PhenoPackage/synthetic_data_for_phenogpt2.json', 'r') as f:
            phenopackage = json.load(f)
        input_data = {}
        for index,p in enumerate(phenopackage):
            input_data[str(index)] = p
    elif 'gsc' in name.lower():
        gsc_gt_dirs = glob.glob('/home/nguyenqm/projects/PhenoGPT2/testing/data/GSC+/*')
        input_data = {}
        for gsc_dir in gsc_gt_dirs:
            file_name = gsc_dir.split("/")[-1].split(".")[0]
            with open(gsc_dir, 'r') as f:
                true_ann = f.readlines()
            true_ann = process_true_ann(true_ann, sep = ' | ')
            input_data[file_name] = {'phenotypes': true_ann}
    elif "id-68" in name.lower():
        id68_gt_dirs = glob.glob('/home/nguyenqm/projects/PhenoGPT2/testing/data/ID-68/*')
        input_data = {}
        for id68dir in id68_gt_dirs:
            file_name = id68dir.split("/")[-1].split(".")[0]
            with open(id68dir, 'r') as f:
                true_ann = f.readlines()
            true_ann = process_true_ann(true_ann, sep = ' | ')
            input_data[file_name] = {'phenotypes': true_ann}
    """
    You should create a dictionary input_data like below:
    input_data = {
        'patid1': {'phenotypes':{'phen1':'HP:XXXXXXX', 'phen2':'HP:XXXXXXX', 'phen3':'HP:XXXXXXX'}}
        'patid2': {'phenotypes':{'phen4':'HP:XXXXXXX', 'phen5':'HP:XXXXXXX', 'phen3':'HP:XXXXXXX'}}
        'patid3': {'phenotypes':{'phen1':'HP:XXXXXXX', 'phen6':'HP:XXXXXXX', 'phen7':'HP:XXXXXXX'}}
        .....
    }
    """
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = '/home/nguyenqm/projects/github_official/PhenoGPT2/data/results/'
    negation_phenotypes, negation_results, negation_empty_ids = process_results(model_dir, name, input_data, negation = False)
    save_data(model_dir, name, negation_phenotypes, negation_results, negation_empty_ids, negation = False)
    print(f"Model: {model_dir}")
    print(f"Dataset: {name}")
    print(f"Negation: {False}")
    print_results(negation_results)
    if negation:
        print()
        print("Now processing for filtered_phenotypes (with negation):")
        negation_phenotypes, negation_results, negation_empty_ids = process_results(model_dir, name, input_data, negation = True)
        save_data(model_dir, name, negation_phenotypes, negation_results, negation_empty_ids, negation = True)
        print(f"Model: {model_dir}")
        print(f"Dataset: {name}")
        print(f"Negation: {True}")
        print_results(negation_results)
if __name__ == "__main__":
    main()