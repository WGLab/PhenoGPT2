from __future__ import annotations

import re
import ast
from typing import List, Optional

import numpy as np
from rapidfuzz import process, fuzz

from .utils import generate_output, generate_output_batch, verify_hpo_mapping
from .formatting_results import extract_json_object


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def exact_or_substring_match(p: str, candidates: List[str]) -> Optional[str]:
    """Case-insensitive exact match."""
    p_l = p.lower()
    for c in candidates:
        if p_l == c.lower():
            return c
    return None


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embedding_match(
    p: str,
    candidates: List[str],
    model,
    threshold: float = 0.90,
) -> Optional[str]:
    """Return the best-matching candidate if similarity >= threshold."""
    if not candidates:
        return None
    p_emb     = model.encode(p, normalize_embeddings=True)
    cand_embs = model.encode(candidates, normalize_embeddings=True)
    sims      = p_emb @ cand_embs.T
    idx       = int(np.argmax(sims))
    return candidates[idx] if sims[idx] >= threshold else None


# ---------------------------------------------------------------------------
# Negation term list and helpers
# ---------------------------------------------------------------------------

negation_check = [
    "no ", "not ", "non ", "none ", "negated", "absent", "negative",
    "without", "never", "intact", "benign", "unremarkable", "free of", "wnl",
]


def segment_bool_flags(s: str, keys=("age", "sex", "ethnicity")):
    """
    Robustly extract true/false values for demographic keys from a raw
    string representation of the negation JSON response.
    """
    key_pos  = {k: re.search(rf"'{re.escape(k)}'\s*:", s) for k in keys}
    BOOL_TOK = re.compile(r"""(["']?)\b(true|false)\b\1""", re.IGNORECASE | re.VERBOSE)

    def next_boundary(curr_key):
        idx = keys.index(curr_key)
        for k2 in keys[idx + 1:]:
            if key_pos[k2]:
                return key_pos[k2].start()
        return len(s)

    out = {}
    for k in keys:
        if not key_pos[k]:
            out[k] = None
            continue
        segment = s[key_pos[k].end():next_boundary(k)]
        matches = list(BOOL_TOK.finditer(segment))
        out[k]  = matches[-1].group(2).lower() == "true" if matches else None
    return out


def contains_negation(text: str, neg_terms: List[str]) -> bool:
    """Return True if any negation term is found in text (lowercased)."""
    text = text.lower()
    return any(term in text for term in neg_terms)


def has_digit_or_period(text: str) -> bool:
    """Return True if text contains a decimal number pattern (e.g. lab values)."""
    return bool(re.search(r"\d+\.\d+", text))


# ---------------------------------------------------------------------------
# Core negation post-processing
# ---------------------------------------------------------------------------

def process_negation(
    output_ans: dict,
    negation_response,
    complete: bool,
    emb_model,
    hpo_verify_threshold: float = 0.6,
) -> dict:
    """
    Apply negation filtering + HPO ID verification to an extracted phenotype record.

    Parameters
    ----------
    output_ans            : dict with 'demographics' and 'phenotypes' keys
    negation_response     : raw string or parsed dict from the negation model
    complete              : whether the extraction was complete
    emb_model             : SentenceTransformer model for embedding operations
    hpo_verify_threshold  : cosine similarity threshold for HPO ID verification

    Steps
    -----
    1. Standard negation filter (evidence + correct flag + type checks).
    2. HPO ID verification: for each phenotype surviving step 1, check that
       its HPO_ID is plausibly matched to the phenotype phrase. If not,
       remove it and record the reason in negation_analysis.

    Returns
    -------
    dict with 'demographics', 'phenotypes', 'filtered_phenotypes',
    'negation_analysis', 'complete' keys.
    """
    temp_dict = {
        "demographics":        {},
        "phenotypes":          {},
        "filtered_phenotypes": {},
        "negation_analysis":   None,
        "complete":            complete,
    }
    demographics = output_ans["demographics"]
    phenotypes   = output_ans["phenotypes"]

    result = extract_json_object(negation_response)
    if not result:
        temp_dict["demographics"]        = output_ans["demographics"]
        temp_dict["phenotypes"]          = output_ans["phenotypes"]
        temp_dict["filtered_phenotypes"] = {}
        temp_dict["negation_analysis"]   = negation_response
        temp_dict["complete"]            = False
        return temp_dict

    demographics_check = result["demographics"]
    phenotypes_check   = result["phenotypes"]
    temp_dict["phenotypes"]        = phenotypes
    temp_dict["negation_analysis"] = result

    # ---- Demographics ----
    fallback = segment_bool_flags(str(demographics_check))

    for field in ("age", "sex", "ethnicity"):
        try:
            correct = demographics_check[field]["correct"]
        except Exception:
            correct = fallback[field]

        if not correct:
            temp_dict["demographics"][field] = (
                "unknown" if isinstance(correct, bool) else demographics[field]
            )
        else:
            temp_dict["demographics"][field] = demographics[field]

    # ---- Step 1: Standard negation filter ----
    all_phenotype_keys = list(phenotypes.keys())
    pred_keys          = list(phenotypes_check.keys())
    negation_survivors = {}   # phenotypes that pass the negation filter

    for p in all_phenotype_keys:
        # 1a. Exact match
        matched_key = exact_or_substring_match(p, pred_keys)

        # 1b. Embedding / fuzzy fallback
        if matched_key is None:
            if emb_model:
                matched_key = embedding_match(p, pred_keys, model=emb_model, threshold=0.90)
            else:
                matched_term, score, _ = process.extractOne(
                    p, pred_keys, scorer=fuzz.token_sort_ratio
                )
                if score >= 70:
                    matched_key = matched_term

        if matched_key:
            try:
                correct_key   = [x for x in phenotypes_check[matched_key].keys() if "correct" in x][0]
                correct_check = ast.literal_eval(phenotypes_check[matched_key][correct_key])
            except Exception:
                correct_check = "true" in str(phenotypes_check[matched_key]).lower()

            if isinstance(phenotypes_check[matched_key], dict):
                evidence_text = (phenotypes_check[matched_key].get("evidence") or "").lower()
            else:
                if correct_check:
                    negation_survivors[p] = phenotypes[p]
                continue

            neg_in_key = contains_negation(matched_key.lower(), negation_check)
            type_str   = phenotypes_check[matched_key].get("type", None)

            if type_str:
                if has_digit_or_period(evidence_text) or has_digit_or_period(matched_key):
                    if type_str == "lab" and "*" in evidence_text:
                        negation_survivors[p] = phenotypes[p]
                        continue
                if (
                    correct_check
                    and not neg_in_key
                    and type_str in ("patient", "patient_indirect", "lab", "imaging")
                ):
                    negation_survivors[p] = phenotypes[p]
                    continue

            if correct_check and not neg_in_key:
                negation_survivors[p] = phenotypes[p]
        else:
            negation_survivors[p] = phenotypes[p]

    # ---- Step 2: HPO ID verification ----
    # Only run when emb_model is available and phenotypes have HPO_ID field
    hpo_removed = {}   # {phenotype: reason} for audit trail

    if emb_model is not None:
        # split into verifiable (have HPO_ID) and pass-through (no HPO_ID)
        to_verify_keys   = []
        to_verify_phrases = []
        to_verify_ids    = []
        passthrough      = {}

        for p, p_data in negation_survivors.items():
            hpo_id = p_data.get("HPO_ID") if isinstance(p_data, dict) else None
            if hpo_id is None:
                passthrough[p] = p_data
            else:
                to_verify_keys.append(p)
                to_verify_phrases.append(p)
                to_verify_ids.append(hpo_id)

        verified_phenotypes = dict(passthrough)  # start with pass-throughs

        if to_verify_keys:
            batch_results = verify_hpo_mapping(
                to_verify_phrases, to_verify_ids, emb_model,
                threshold=hpo_verify_threshold,
            )
            for p, p_data, (is_valid, reason) in zip(
                to_verify_keys, 
                [negation_survivors[k] for k in to_verify_keys],
                batch_results
            ):
                if is_valid:
                    verified_phenotypes[p] = p_data
                else:
                    hpo_removed[p] = reason

        temp_dict["filtered_phenotypes"] = verified_phenotypes
    else:
        # No emb_model available — skip HPO verification, keep all survivors
        temp_dict["filtered_phenotypes"] = negation_survivors

    # ---- Attach HPO verification audit to negation_analysis ----
    if hpo_removed:
        if isinstance(temp_dict["negation_analysis"], dict):
            temp_dict["negation_analysis"]["hpo_verification_removed"] = hpo_removed
        else:
            temp_dict["negation_analysis"] = {
                "raw":                      temp_dict["negation_analysis"],
                "hpo_verification_removed": hpo_removed,
            }

    return temp_dict


# ---------------------------------------------------------------------------
# Detection functions — accept VllmClient
# ---------------------------------------------------------------------------

def negation_detection(
    model,
    chunk: str,
    data_point: dict,
    max_new_tokens: int = 16384,
) -> str:
    """Single-sample negation detection via VllmClient."""
    data_point = data_point.copy()
    data_point["clinical_note"] = chunk.lower()
    return generate_output(
        model,
        data_point,
        max_new_tokens=max_new_tokens,
        temperature=0.4,
        negation_detect=True,
    )


def negation_detection_batch(
    model,
    chunks: List[str],
    data_points: List[dict],
    max_new_tokens: int = 5000,
) -> List[str]:
    """Batched negation detection via VllmClient."""
    batched_inputs = []
    for chunk, dp in zip(chunks, data_points):
        dp2 = dp.copy()
        dp2["clinical_note"] = chunk.lower()
        batched_inputs.append(dp2)

    return generate_output_batch(
        model,
        batched_inputs,
        temperature=0.3,
        negation_detect=True,
        max_new_tokens=max_new_tokens,
    )