#import spacy, medspacy
#from medspacy.context import ConText, ConTextRule
#from medspacy.ner import TargetRule
#from spacy.tokens import Doc
from ast import literal_eval
from .utils import generate_output
from .formatting_results import *
from rapidfuzz import process, fuzz
import ast
# nlp = spacy.blank("en")
# def set_target_rules( phenotype_list):
#     return [TargetRule(phen, "PHENOTYPE") for phen in phenotype_list]
# def include_ent(ent):
#     if ent._.is_negated:
#         return False
#     if ent._.is_uncertain:
#         return False
# #         if ent._.is_historical:
# #             return False
#     if ent._.is_hypothetical:
#         return False
#     if ent._.is_family:
#         return False
#     return True
# def remove_negation_by_medspacy(paragraph, phenotypes):
#     nlp = medspacy.load()
#     target_matcher = nlp.get_pipe("medspacy_target_matcher")
#     texts = paragraph.split(".")
#     texts = [t.strip() for t in texts if len(t) > 0]
#     target_rules = set_target_rules(phenotypes)
#     target_matcher.add(target_rules)
#     docs = list(nlp.pipe(texts))
#     positive_phens = []
#     for doc in docs:
#         for ent in doc.ents:
#             if include_ent(ent) and str(ent) not in positive_phens:
#                 positive_phens.append(str(ent))
#     return(positive_phens)
def exact_or_substring_match(p, candidates):
    """
    Check exact or substring match (case-insensitive).
    """
    p_l = p.lower()
    for c in candidates:
        c_l = c.lower()
        if p_l == c_l:# or p_l in c_l or c_l in p_l:
            return c
    return None
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def embedding_match(p, candidates, model, threshold=0.90):
    """
    Return best matching candidate if similarity >= threshold, else None
    """
    if not candidates:
        return None

    p_emb = model.encode(p, normalize_embeddings=True)
    cand_embs = model.encode(candidates, normalize_embeddings=True)

    sims = p_emb @ cand_embs.T
    idx = int(np.argmax(sims))
    if sims[idx] >= threshold:
        return candidates[idx]
    return None
negation_check = ['no ', 'not ', 'non ', 'none ', 'negated', 'absent', 'negative', 'without', 'never', 'intact',
                 'benign', 'unremarkable', 'free of', 'wnl']
def segment_bool_flags(s: str, keys=("age", "sex", "ethnicity")):
    # Find key positions (start indices)
    key_pos = {k: re.search(rf"'{re.escape(k)}'\s*:", s) for k in keys}

    # token pattern: true/false, optionally quoted, case-insensitive
    BOOL_TOK = re.compile(r"""(["']?)\b(true|false)\b\1""", re.IGNORECASE | re.VERBOSE)

    def next_boundary(curr_key):
        """Return index of the next existing key after curr_key, else end-of-string."""
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

        start = key_pos[k].end()
        end = next_boundary(k)
        segment = s[start:end]

        matches = list(BOOL_TOK.finditer(segment))
        if not matches:
            out[k] = None
        else:
            # pick the LAST boolean token in the segment (most "local"/closest to boundary)
            out[k] = matches[-1].group(2).lower() == "true"

    return out

def contains_negation(text, neg_terms):
    text = text.lower()
def has_digit_or_period(text):
    pattern = re.compile(r'\d+\.\d+')
    return bool(pattern.search(text))
def process_negation(output_ans, negation_response, complete, emb_model):
    temp_dict = {'demographics': {}, 'phenotypes':{}, 'filtered_phenotypes': {}, 'negation_analysis': None, 'complete': complete}
    demographics = output_ans['demographics']
    phenotypes = output_ans['phenotypes']
    result = extract_json_object(negation_response)
    if not result:
        temp_dict['demographics'] = output_ans['demographics']
        temp_dict['phenotypes'] = output_ans['phenotypes']
        temp_dict['filtered_phenotypes'] = {}
        temp_dict['negation_analysis'] = negation_response
        temp_dict['complete'] = False 
    demographics_check = result['demographics']
    phenotypes_check = result['phenotypes']
    temp_dict['phenotypes'] = phenotypes
    temp_dict['negation_analysis'] = result
    # ---------- demographics ----------
    demographics_check_fallback = segment_bool_flags(str(demographics_check))
    try:
        age_correct = demographics_check['age']['correct']
    except:
        age_correct = demographics_check_fallback['age']
    if not age_correct:
        if isinstance(age_correct, bool):
            temp_dict['demographics']['age'] = 'unknown'
        else:
            temp_dict['demographics']['age'] = demographics['age']
    else:
        temp_dict['demographics']['age'] = demographics['age']

    try:
        sex_correct = demographics_check['sex']['correct']
    except:
        sex_correct = demographics_check_fallback['sex']
    if not sex_correct:
        if isinstance(sex_correct, bool):
            temp_dict['demographics']['sex'] = 'unknown'
        else:
            temp_dict['demographics']['sex'] = demographics['sex']
    else:
        temp_dict['demographics']['sex'] = demographics['sex']

    try:
        ethnicity_correct = demographics_check['ethnicity']['correct']
    except:
        ethnicity_correct = demographics_check_fallback['ethnicity']
    if not ethnicity_correct:
        if isinstance(ethnicity_correct, bool):
            temp_dict['demographics']['ethnicity'] = 'unknown'
        else:
            temp_dict['demographics']['ethnicity'] = demographics['ethnicity']
    else:
        temp_dict['demographics']['ethnicity'] = demographics['ethnicity']

    # ---------- phenotypes ----------
    all_phenotypes = list(phenotypes.keys())
    pred_keys = list(phenotypes_check.keys())
    for p in all_phenotypes:
        matched_key = None

        # 1️⃣ exact / substring match
        matched_key = exact_or_substring_match(p, pred_keys)

        # 2️⃣ embedding fallback (only if not matched)
        if matched_key is None:
            if emb_model:
                matched_key = embedding_match(
                    p,
                    pred_keys,
                    model=emb_model,   # <-- your LLM / embedding model
                    threshold=0.90
                )
            else:
                matched_term, score, _ = process.extractOne(
                    p,
                    pred_keys,
                    scorer=fuzz.token_sort_ratio
                )
                if score >= 70:
                    matched_key = matched_term
        if matched_key:
            try:
                correct_key = [
                    x for x in phenotypes_check[matched_key].keys()
                    if 'correct' in x
                ][0]
                correct_check = ast.literal_eval(phenotypes_check[matched_key][correct_key])
            except:
                if 'true' in str(phenotypes_check[matched_key]).lower():
                    correct_check = True
                else:
                    correct_check = False
            if isinstance(phenotypes_check[matched_key], dict):
                evidence_text = phenotypes_check[matched_key].get('evidence', '')
                if evidence_text:
                    evidence_text = evidence_text.lower()
                else:
                    evidence_text = ''
            else:
                if correct_check:
                    temp_dict['filtered_phenotypes'][p] = phenotypes[p]
                continue
            matched_key_lower = matched_key.lower()

            # ---- negation checks ----
            #neg_in_evidence = contains_negation(evidence_text, negation_check)
            neg_in_key = contains_negation(matched_key_lower, negation_check)

            # ---- lab special rule ----
            check_lab = phenotypes_check[matched_key].get('type', None)
            if check_lab:
                if (has_digit_or_period(evidence_text) or has_digit_or_period(matched_key)):
                    if check_lab == 'lab' and ('*' in evidence_text):
                        temp_dict['filtered_phenotypes'][p] = phenotypes[p]
                        continue
            if (
                correct_check
                and not neg_in_key
            ):
                temp_dict['filtered_phenotypes'][p] = phenotypes[p]

        else:
            temp_dict['filtered_phenotypes'][p] = phenotypes[p]
    return temp_dict
def negation_detection(model, tokenizer, chunk, data_point, device, max_new_tokens = 16384):
    #positive_phenotypes = remove_negation_by_medspacy(text.lower(), phenotypes)
    data_point = data_point.copy()
    data_point['clinical_note'] = chunk.lower()
    negation_response = generate_output(model, tokenizer, data_point, temperature = 0.4, negation_detect = True, max_new_tokens = max_new_tokens, device = device)
    return negation_response
