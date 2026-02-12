import sys
from .formatting_results import *
from .negation import *
import networkx as nx
import json, pickle, glob, ast
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from copy import deepcopy
from tqdm import tqdm
import os, ast
# Load the model
emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
def load_hpo_graph(obo_path: str) -> nx.DiGraph:
    """
    Load HPO ontology as a directed graph.
    Edges are child -> parent.
    """
    G = nx.DiGraph()

    current_id = None
    parents = []

    def flush():
        nonlocal current_id, parents
        if current_id:
            G.add_node(current_id)
            for p in parents:
                G.add_edge(current_id, p)
        current_id = None
        parents = []

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                flush()
            elif line.startswith("id: HP:"):
                current_id = line.split("id:")[1].strip()
            elif line.startswith("is_a: HP:"):
                parent = line.split("is_a:")[1].split("!")[0].strip()
                parents.append(parent)

        flush()

    return G
def compute_depths(G: nx.DiGraph, root: str = "HP:0000001"):
    """
    Compute shortest depth from root to all nodes.
    """
    depths = {root: 0}
    queue = [root]

    while queue:
        node = queue.pop(0)
        for child in G.predecessors(node):  # reverse edge: parent -> child
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)

    return depths
def ancestors(G: nx.DiGraph, node: str):
    """Return all ancestors including itself."""
    return set(nx.descendants(G, node)) | {node}


def lowest_common_ancestor(G, node1, node2, depths):
    common = ancestors(G, node1) & ancestors(G, node2)
    if not common:
        return None
    return max(common, key=lambda n: depths.get(n, -1))
def wu_palmer_similarity(G, depths, t1: str, t2: str) -> float:
    if t1 not in depths or t2 not in depths:
        return 0.0

    lca = lowest_common_ancestor(G, t1, t2, depths)
    if lca is None:
        return 0.0

    d_lca = depths[lca]
    d_t1 = depths[t1]
    d_t2 = depths[t2]

    return (2.0 * d_lca) / (d_t1 + d_t2)
def compute_hp0000118_descendants(G: nx.DiGraph) -> set[str]:
    """
    Precompute all descendants of HP:0000118.
    """
    # descendants() walks *down* → need reversed graph
    Grev = G.reverse(copy=False)
    return nx.descendants(Grev, 'HP:0000118')

G = load_hpo_graph("/home/nguyenqm/projects/MedicalDBs/HPO/hp.obo")
depths = compute_depths(G)
HP0000118_DESC = compute_hp0000118_descendants(G)

def is_descendant_of_hp0000118(hpo_id: str) -> bool:
    return hpo_id in HP0000118_DESC

def process_true_ann(true_ann, sep = ' | '):
    if len(true_ann) > 0:
        true_ann_split = [t.strip().split(sep) for t in true_ann]
        exist_check = []
        true_ann = {}
        for t in true_ann_split:
            if t[1] not in exist_check:
                true_ann[t[0]] = t[1]
        return true_ann
    else:
        return {}
def process_true_ann_from_dict(ann_dir):
    with open(ann_dir, 'rb') as f:
        data = pickle.load(f)
    ground_truth = {}
    for k,v in data.items():
        ground_truth[k] = {phen.lower():phen_dict['HPO_ID'] for phen,phen_dict in v['filtered_phenotypes'].items() if is_descendant_of_hp0000118(phen_dict['HPO_ID'])}
    return ground_truth
def generate_id68_labels(dataset_dirs):
    all_dict = {}
    for dataset_dir in dataset_dirs:
        file_name = dataset_dir.split("/")[-1].split(".")[0]
        with open(dataset_dir, 'r') as f:
            texts = f.readlines()
        texts = [t.strip() for t in texts if len(t.strip()) > 0]
        df = pd.DataFrame([x.replace('ul\t','').strip().split('\t') for x in texts])
        if len(df.columns) == 5:
            df.columns = ['start','end', 'phenotype','hpo','filter']
            df.loc[pd.isna(df['filter']), 'relation'] = 'Pos'
        else:
            df.columns = ['start','end', 'phenotype','hpo']
            df['filter'] = 'Pos'
        df = df.loc[df['filter'] != 'Neg', ['phenotype','hpo']]
        all_dict[file_name] = df.set_index('phenotype')['hpo'].to_dict()
    return all_dict
def generate_gsc_labels(dataset_dirs):
    all_dict = {}
    for dataset_dir in dataset_dirs:
        file_name = dataset_dir.split("/")[-1].split(".")[0]
        with open(dataset_dir, 'r') as f:
            texts = f.readlines()
        texts = [t.strip().split("\t")[1].split(" | ") for t in texts if len(t.strip()) > 0]
        df = pd.DataFrame(texts, columns = ['hpo', 'phenotype'])
        df['hpo'] = df['hpo'].str.replace("_", ":")
        all_dict[file_name] = df.set_index('phenotype')['hpo'].to_dict()
    return all_dict
def golden_labels(dataset_name):
    """
    The format should be like this:
    {
    'pat1': 
        {
            'phen1':'hpo1',
            'phen2':'hpo2',
            'phen3':'hpo3'
        },
    'pat2': 
        {
            'phen4':'hpo4',
            'phen5':'hpo5',
            'phen6':'hpo6'
        },
    }
    
    """
    if dataset_name == 'arcus':
        ground_truth = process_true_ann_from_dict('/home/nguyenqm/projects/github/PhenoGPT2/testing_data/Arcus/clean_annotations.pkl')
    elif dataset_name == 'phenopackets':
        with open('/home/nguyenqm/projects/MedicalDBs/PhenoPacket/synthetic_phenopacket_dataset.pkl', 'rb') as f:
            gt_list = pickle.load(f)
        ground_truth = {}
        for sample in gt_list:
            ground_truth[sample['pid']] = {}
            for k,v in sample['phenotypes'].items():
                if is_descendant_of_hp0000118(v):
                    ground_truth[sample['pid']][k] = v
    elif dataset_name == 'id-68':
        ground_truth = process_true_ann_from_dict('/home/nguyenqm/projects/github/PhenoGPT2/testing_data/ID-68/clean_annotations.pkl')
    elif dataset_name == 'gsc':
        ground_truth = process_true_ann_from_dict('/home/nguyenqm/projects/github/PhenoGPT2/testing_data/GSC+/clean_annotations.pkl')
    elif dataset_name == 'phenopackets_typo':
        with open('/home/nguyenqm/projects/PhenoGPT2/evaluations/robustness/phenopackets_data_with_typos.json', 'r') as f:
            gt_list = json.load(f)
        ground_truth = {}
        for sample in gt_list:
            ground_truth[sample['pid']] = {}
            for k,v in sample['phenotypes'].items():
                if is_descendant_of_hp0000118(v['HPO_ID']):
                    ground_truth[sample['pid']][k] = v['HPO_ID']
    final_gt = {}
    
    for k,v in ground_truth.items():
        existing_hpo = []
        final_gt[k] = {}
        for phen, hpo in v.items():
            if hpo not in existing_hpo:
                final_gt[k][phen] = hpo
                existing_hpo.append(hpo)
    return final_gt

def match_predictions(
    ground_truth,
    pred_phens,
    G,
    depths,
    embed_model=None,
    wu_threshold=0.8,
    text_threshold=0.8,
):
    """
    Returns:
      matched_pred: dict
      unmatched_pred: dict
      uncovered_gt: dict
      stats: dict
    """
    pred_phens = {k:v for k,v in pred_phens.items() if is_descendant_of_hp0000118(v)}
    matched_pred = {}
    used_gt = set()
    matched_pred_texts = set()
    if len(ground_truth) == 0:
        matched_pred = {}
        unmatched_pred = pred_phens
        uncovered_gt = ground_truth
        if len(pred_phens) == 0:
            stats = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": round(1, 4),
                "recall": round(1, 4),
                "f1": round(1, 4),
                "accuracy": round(1, 4),
                "avg_wu_palmer": round(1, 4),
            }
        else:
            stats = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": round(0, 4),
                "recall": round(0, 4),
                "f1": round(0, 4),
                "accuracy": round(0, 4),
                "avg_wu_palmer": round(0, 4),
            }
        return matched_pred, unmatched_pred, uncovered_gt, stats
    if len(pred_phens) == 0:
        matched_pred = {}
        unmatched_pred = pred_phens
        uncovered_gt = ground_truth
        stats = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": round(0, 4),
                "recall": round(0, 4),
                "f1": round(0, 4),
                "accuracy": round(0, 4),
                "avg_wu_palmer": round(0, 4),
            }
        return matched_pred, unmatched_pred, uncovered_gt, stats
    # Pre-embed texts if embedding model is provided
    gt_embeddings = {}
    pr_embeddings = {}
    if embed_model:
        for gt_text in ground_truth:
            gt_embeddings[gt_text] = embed_model.encode([gt_text])
        for pred_text in pred_phens:
            pr_embeddings[pred_text] = embed_model.encode([pred_text])

    tp = 0
    unmatched_preds = set(pred_phens.keys())
    unextracted_gts = set(ground_truth.keys())

    for pred_text in list(unmatched_preds):
        pred_hpo = pred_phens[pred_text]

        best_match = None
        best_wu = 0.0
        for gt_text in unextracted_gts:
            wu = wu_palmer_similarity(G, depths, pred_hpo, ground_truth[gt_text])
            if pred_hpo == ground_truth[gt_text]:
                best_match = gt_text
                best_wu = wu
                break
            if wu < wu_threshold:
                continue

            if embed_model:
                sim_text = embed_model.similarity(
                    pr_embeddings[pred_text],
                    gt_embeddings[gt_text]
                )[0]
                if sim_text < text_threshold:
                    continue

            if wu > best_wu:
                best_wu = wu
                best_match = gt_text

        if best_match:
            tp += 1
            unmatched_preds.remove(pred_text)
            unextracted_gts.remove(best_match)

            matched_pred[pred_text] = {
                "pred_hpo": pred_hpo,
                "gt_text": best_match,
                "gt_hpo": ground_truth[best_match],
                "wu_palmer": round(best_wu, 4),
            }

    for pred_text in list(unmatched_preds):
        pred_hpo = pred_phens[pred_text]

        best_match = None
        best_wu = 0.0

        for gt_text in unextracted_gts:
            wu = wu_palmer_similarity(G, depths, pred_hpo, ground_truth[gt_text])
            if wu >= wu_threshold and wu > best_wu:
                best_wu = wu
                best_match = gt_text

        if best_match:
            tp += 1
            unmatched_preds.remove(pred_text)
            unextracted_gts.remove(best_match)

            matched_pred[pred_text] = {
                "pred_hpo": pred_hpo,
                "gt_text": best_match,
                "gt_hpo": ground_truth[best_match],
                "wu_palmer": round(best_wu, 4),
            }

    fp = len(unmatched_preds)
    fn = len(unextracted_gts)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = tp / len(ground_truth) if ground_truth else 0.0

    avg_wu = (
        sum(v["wu_palmer"] for v in matched_pred.values()) / tp
        if tp else 0.0
    )

    stats = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "avg_wu_palmer": round(avg_wu, 4),
    }

    return matched_pred, list(unmatched_preds), list(unextracted_gts), stats
def generate_phenotagger_annotations(phenotagger_dirs):
    all_dict = {}
    if 'phenopacket_typo' in phenotagger_dirs[0]:
        with open('/home/nguyenqm/projects/PhenoGPT2/evaluations/robustness/phenopackets_data_with_typos.json', 'r') as f:
            gt_list = json.load(f)
        index2name = {str(index):x['pid'] for index, x in enumerate(gt_list)}
    elif 'phenopacket_full' in phenotagger_dirs[0]:
        with open('/home/nguyenqm/projects/MedicalDBs/PhenoPacket/synthetic_phenopacket_dataset.pkl', 'rb') as f:
            gt_list = pickle.load(f)
        index2name = {str(index):x['pid'] for index, x in enumerate(gt_list)}
    else:
        index2name = {}
    for phenotagger_dir in phenotagger_dirs:
        file_name = phenotagger_dir.split("/")[-1].split(".")[0]
        if len(index2name) > 0:
            file_name = index2name[file_name]
        with open(phenotagger_dir, 'r') as f:
            texts = f.readlines()
        texts = [t.strip() for t in texts if len(t.strip()) > 0]
        all_dict[file_name] = {'text': {}}
        if len(texts) > 0:
            df = pd.DataFrame([x.replace('ul\t','').strip().split('\t')[3:] for x in texts[2:]], columns = ['phenotype','category','hpo','score','filter'])
            all_dict[file_name]['text']['phenotypes'] = df.set_index('phenotype')['hpo'].to_dict()
            df = df.loc[df['filter'] == 'positive', ['phenotype','hpo']]
            all_dict[file_name]['text']['filtered_phenotypes'] = df.set_index('phenotype')['hpo'].to_dict()
    return all_dict

def evaluate(all_configs):
    #all_configs = ['phenogpt2_llama_ehr_8b_ft_nofilter']
    done_list = []
    full_metrics = {}
    full_extractions = {}
    full_remaining = {}
    for filtered_phen in [True, False]:
        if filtered_phen:
            phen_field = 'filtered_phenotypes'
        else:
            phen_field = 'phenotypes'
        for config in all_configs:
            all_datasets = glob.glob(f'/home/nguyenqm/projects/github/PhenoGPT2/{config}/evaluations/*')
            for out_dir in all_datasets:
                if "mimic" in out_dir or "ID-68_check" in out_dir or "CHOP_IMAGES_PUBMIND" in out_dir:#("PhenoPackets/" in (out_dir+"/")):
                    continue
                print(out_dir)
                data_dir = out_dir + "/phenogpt2_rep0.pkl"
                with open(data_dir, 'rb') as f :
                    all_dict = pickle.load(f)
                # load ground truth
                if 'arcus' in out_dir.lower():
                    dataset_name = 'arcus'
                elif 'phenopackets_typo' in out_dir.lower():
                    dataset_name = 'phenopackets_typo'
                elif 'phenopackets_full' in out_dir.lower():
                    dataset_name = 'phenopackets'
                elif 'phenopackets' in out_dir.lower():
                    dataset_name = 'phenopackets'
                elif 'gsc' in out_dir.lower():
                    dataset_name = 'gsc'
                else:
                    dataset_name = 'id-68'
                ground_truth = golden_labels(dataset_name)
                # generate evaluation results
                all_metrics = {}
                all_extractions = {}
                all_remaining = {}
                for pat_id in tqdm(ground_truth.keys()):
                    #pred_phens_raw = {k:v['HPO_ID'] for k,v in all_dict[pat_id]['text'][phen_field].items() if "HPO_ID" in v}
                    existing_check = []
                    pred_phens = {}
                    if phen_field not in all_dict[pat_id]['text']:
                        try:
                            pred_phens = ast.literal_eval(all_dict[pat_id]['text']['error_response'])['phenotypes']
                        except:
                            try:
                                pred_phens = json.loads(all_dict[pat_id]['text']['error_response'])['phenotypes']
                            except:
                                pred_phens = {}
                    else:
                        if isinstance(all_dict[pat_id]['text'][phen_field], str):
                            final_response = all_dict[pat_id]['text']
                            negation_response = all_dict[pat_id]['text']['negation_analysis']
                            complete_check = True
                            all_dict[pat_id]['text'] = process_negation(final_response, negation_response, complete_check, emb_model)
                        for k,v in all_dict[pat_id]['text'][phen_field].items():
                            if 'HPO_ID' in v and v['HPO_ID'] not in existing_check:
                                pred_phens[k] = v['HPO_ID']
                                existing_check.append(v['HPO_ID'])
                    matched_pred, unmatched_pred, uncovered_gt, metrics = match_predictions(
                        ground_truth[pat_id],
                        pred_phens,
                        G,
                        depths,
                        embed_model=emb_model,   # or Qwen-3 embedding fn
                        wu_threshold=0.8,
                        text_threshold=0.8
                    )
                    all_metrics[pat_id] = metrics
                    all_extractions[pat_id] = matched_pred
                    all_remaining[pat_id] = {'unmatched': unmatched_pred, 'unextracted': uncovered_gt}
                full_metrics[(config,dataset_name,phen_field)] = all_metrics
                full_extractions[(config,dataset_name,phen_field)] = all_extractions
                full_remaining[(config,dataset_name,phen_field)] = all_remaining
                with open(f'{out_dir}/all_metrics_{phen_field}.json', 'w') as f:
                    json.dump(all_metrics, f, indent = 4)
                with open(f'{out_dir}/all_extractions_{phen_field}.json', 'w') as f:
                    json.dump(all_extractions, f, indent = 4)
                with open(f'{out_dir}/all_remaining_{phen_field}.json', 'w') as f:
                    json.dump(all_remaining, f, indent = 4)
def main():
    all_configs = ['phenogpt2_llama_ehr_8b_ft_nofilter']
    evaluate(all_configs)

if __name__ == "__main__":
    main()
