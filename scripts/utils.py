from __future__ import annotations

import json
import glob
import os
import pickle
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np

from .prompting import generate_prompt, prompt_negation

# ---------------------------------------------------------------------------
# HPO database (loaded once at import time)
# ---------------------------------------------------------------------------

script_dir   = Path(__file__).resolve().parent
hpo_db_file  = script_dir.parent / "data" / "hpo_added_tokens.json"
with open(hpo_db_file, "r") as _f:
    hpo_db = json.load(_f)

# all_phenotypes : list of phenotype name strings (keys of hpo_db)
all_phenotypes = list(hpo_db.keys())

# Pre-computed HPO name → canonical name lookup (value of hpo_db is the HP: id)
# We also build a reverse map: HPO_ID → list of canonical phenotype names
_hpo_id_to_names: dict[str, list[str]] = {}
for _name, _hpo_id in hpo_db.items():
    _hpo_id_to_names.setdefault(_hpo_id, []).append(_name)


# ---------------------------------------------------------------------------
# Embedding-based HPO matching (shared by vision pipeline + negation checker)
# ---------------------------------------------------------------------------

# Module-level cache so phenotype embeddings are only computed once
_phenotype_embeddings = None
_phenotype_names      = None   # aligned list of names


def _get_phenotype_embeddings(emb_model):
    """
    Lazily compute and cache embeddings for all HPO phenotype names.
    Returns (embeddings ndarray, names list).
    """
    global _phenotype_embeddings, _phenotype_names
    if _phenotype_embeddings is None:
        _phenotype_names      = all_phenotypes
        _phenotype_embeddings = emb_model.encode(
            _phenotype_names,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return _phenotype_embeddings, _phenotype_names


def embed_and_match_hpo(
    phrase: str,
    emb_model,
    threshold: float = 0.8,
) -> Optional[str]:
    """
    Find the best-matching HPO ID for a phenotype phrase using cosine similarity.

    Parameters
    ----------
    phrase    : free-text phenotype phrase
    emb_model : SentenceTransformer model
    threshold : minimum cosine similarity to accept a match

    Returns
    -------
    HPO ID string (e.g. 'HP:0000123') if similarity >= threshold, else None.
    """
    embs, names = _get_phenotype_embeddings(emb_model)
    q_emb       = emb_model.encode(phrase, normalize_embeddings=True)
    sims        = q_emb @ embs.T
    idx         = int(np.argmax(sims))
    if sims[idx] >= threshold:
        return hpo_db[names[idx]]
    return None


def verify_hpo_mapping(
    phenotype_phrases: list[str],
    hpo_ids: list[str],
    emb_model,
    threshold: float = 0.6,
) -> list[tuple[bool, str]]:
    """
    Batched version of verify_hpo_mapping.
    Encodes all phrases in one call and all canonical names in one call.
    """
    # collect all canonical names and track offsets
    all_names   = []
    offsets     = []   # (start, end) index into all_names per phrase
    results_pre = []   # store early exits (hpo_id not found)

    for phrase, hpo_id in zip(phenotype_phrases, hpo_ids):
        canonical_names = _hpo_id_to_names.get(hpo_id, [])
        if not canonical_names:
            offsets.append(None)
            results_pre.append((False, f"HPO ID {hpo_id} not found in database."))
        else:
            start = len(all_names)
            all_names.extend(canonical_names)
            offsets.append((start, len(all_names)))
            results_pre.append(None)

    # batch encode all phrases and all canonical names in two calls
    phrase_embs = emb_model.encode(
        phenotype_phrases, normalize_embeddings=True, show_progress_bar=False
    )
    name_embs = emb_model.encode(
        all_names, normalize_embeddings=True, show_progress_bar=False
    ) if all_names else np.array([])

    results = []
    for i, (phrase, hpo_id) in enumerate(zip(phenotype_phrases, hpo_ids)):
        if results_pre[i] is not None:
            results.append(results_pre[i])
            continue

        start, end      = offsets[i]
        canonical_names = _hpo_id_to_names[hpo_id]
        sims            = phrase_embs[i] @ name_embs[start:end].T
        best_idx        = int(np.argmax(sims))
        best_sim        = float(sims[best_idx])
        best_name       = canonical_names[best_idx]

        if best_sim >= threshold:
            results.append((True, f"Matched '{best_name}' (sim={best_sim:.2f})"))
        else:
            results.append((False,
                f"The extracted HPO ID {hpo_id} is mapped to '{best_name}' "
                f"(sim={best_sim:.2f} < {threshold}), so this is an incorrect "
                f"mapping and has been removed."
            ))

    return results

# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def read_one_file(input_file):
    if isinstance(input_file, list):
        for i, dt in enumerate(input_file):
            if "input" in input_file[i]:
                input_file[i]["clinical_note"] = input_file[i].pop("input")
            if "image" not in input_file[i]:
                input_file[i]["image"] = ""
            input_file[i]["pid"] = input_file[i].pop("id")
        return {dt["pid"]: dt for dt in input_file}

    elif os.path.isfile(input_file):
        if ".pkl" in input_file:
            with open(input_file, "rb") as f:
                input_dict = pickle.load(f)
        elif ".json" in input_file:
            with open(input_file, "r") as f:
                input_dict = json.load(f)
        else:
            file_name = input_file.split("/")[-1].split(".")[0]
            if any(x in input_file for x in ["png", "jpg", "jpeg"]):
                input_text = np.nan
                image_dir  = input_file
            else:
                with open(input_file, "r") as f:
                    input_data = f.readlines()
                input_text = "\t".join(d.strip() for d in input_data)
                image_dir  = np.nan
            return {file_name: {"clinical_note": input_text,
                                "image": image_dir,
                                "pid": file_name}}

        if isinstance(input_dict, list):
            if "clinical_note" not in input_dict[0] and "input" in input_dict[0]:
                for i in range(len(input_dict)):
                    input_dict[i]["clinical_note"] = input_dict[i].pop("input")
            assert "clinical_note" in input_dict[0], \
                "The given data is missing 'clinical_note' key."
            return {
                (dt["pid"] if "pid" in dt else dt["pmid"]): dt
                for dt in input_dict
            }

        elif isinstance(input_dict, dict):
            if "pid" in input_dict:
                input_dict_copy = deepcopy(input_dict)
                pid = input_dict_copy["pid"]
                input_dict = {pid: {}}
                for k, v in input_dict_copy.items():
                    input_dict[pid][k] = v
            elif "pmid" in input_dict:
                input_dict_copy = deepcopy(input_dict)
                pid = input_dict_copy["pmid"]
                input_dict = {pid: {}}
                for k, v in input_dict_copy.items():
                    input_dict[pid][k] = v
            else:
                for v in input_dict.values():
                    assert "clinical_note" in v, \
                        "The given data is missing 'clinical_note' key."
                    break
            return input_dict

        else:
            raise ImportError("This file is not in correct format (list or dictionary)")
    else:
        raise ImportError("This is not a file")


def read_input(input_file):
    if isinstance(input_file, list) or os.path.isfile(input_file):
        return read_one_file(input_file)
    input_dict = {}
    for f in glob.glob(input_file + "/*"):
        input_dict.update(read_one_file(f))
    return input_dict


# ---------------------------------------------------------------------------
# Generation helpers — backed by VllmClient
# ---------------------------------------------------------------------------

def generate_output(
    client,
    data_point,
    max_new_tokens: int = 5000,
    temperature: float = 0.5,
    negation_detect: bool = False,
    task: str = "running",
):
    """Single-sample inference via VllmClient."""
    if negation_detect:
        messages       = prompt_negation(data_point)
        max_new_tokens = min(32768, max_new_tokens * 2)
    else:
        messages = generate_prompt(data_point)

    system = messages[0]["content"]
    user   = messages[1]["content"]

    return client.chat(
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_new_tokens,
        task=task,
    )


def generate_output_batch(
    client,
    data_points: List,
    max_new_tokens: int = 5000,
    temperature: float = 0.5,
    negation_detect: bool = False,
    task: str = "running",
) -> List:
    """Batched inference via VllmClient."""
    if not data_points:
        return []

    systems, users  = [], []
    _max_new_tokens = max_new_tokens
    for dp in data_points:
        if negation_detect:
            messages        = prompt_negation(dp)
            _max_new_tokens = min(32768, max_new_tokens * 2)
        else:
            messages = generate_prompt(dp)
        systems.append(messages[0]["content"])
        users.append(messages[1]["content"])

    system = systems[0]

    return client.chat_batch(
        system=system,
        users=users,
        temperature=temperature,
        max_tokens=_max_new_tokens,
        task=task,
    )


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_outputs(data: dict) -> dict:
    """Combine all chunk results for one clinical note into a single record."""
    pheno_dict                 = {}
    filtered_pheno_dict        = {}
    evidence_demographics_dict = {}
    evidence_phenotypes_dict   = {}
    demographics               = {}
    check                      = 0
    has_valid_chunk            = False
    last_error                 = None

    for index, (tid, data_dict) in enumerate(data.items()):
        if "error_response" not in data_dict:
            has_valid_chunk = True

            if index == check and len(data_dict) > 0:
                demographics = data_dict["demographics"]
                try:
                    evidence_demographics_dict = \
                        data_dict["negation_analysis"]["demographics"]
                except Exception:
                    pass

            for field in ("age", "sex", "ethnicity"):
                if field not in data_dict["demographics"]:
                    data_dict["demographics"][field] = "unknown"

            for field in ("age", "sex", "ethnicity"):
                if (
                    len(demographics) > 0
                    and isinstance(data_dict["demographics"].get(field), str)
                    and str(demographics.get(field, "")).lower() == "unknown"
                    and len(data_dict) > 0
                ):
                    demographics[field] = data_dict["demographics"][field]
                    try:
                        evidence_demographics_dict[field] = \
                            data_dict["negation_analysis"]["demographics"][field]
                    except Exception:
                        pass

            try:
                pheno_dict.update(data_dict["phenotypes"])
            except Exception:
                pass
            try:
                filtered_pheno_dict.update(data_dict["filtered_phenotypes"])
            except Exception:
                pass
            try:
                evidence_phenotypes_dict.update(
                    data_dict["negation_analysis"]["phenotypes"]
                )
            except Exception:
                pass

            if (
                not evidence_demographics_dict
                and "negation_analysis" in data_dict
                and isinstance(data_dict["negation_analysis"], dict)
                and "demographics" in data_dict["negation_analysis"]
            ):
                evidence_demographics_dict = \
                    data_dict["negation_analysis"]["demographics"]
        else:
            last_error = data_dict["error_response"]

        if not demographics:
            demographics = {"age": "unknown", "sex": "unknown", "ethnicity": "unknown"}

    negation_dict = {
        "demographics": evidence_demographics_dict,
        "phenotypes":   evidence_phenotypes_dict,
    }

    def _dedup(d):
        seen, out = [], {}
        for k, v in d.items():
            if isinstance(v, dict) and "HPO_ID" in v and v["HPO_ID"] not in seen:
                out[k] = v
                seen.append(v["HPO_ID"])
        return out

    result = {
        "demographics":        demographics,
        "phenotypes":          _dedup(pheno_dict),
        "filtered_phenotypes": _dedup(filtered_pheno_dict),
        "negation_analysis":   negation_dict,
    }
    if not has_valid_chunk:
        result["error_response"] = last_error
    return result


def count_tokens(text: str, tokenizer) -> int:
    """Approximate token count using a HuggingFace tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))