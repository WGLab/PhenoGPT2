
import glob, os, re, pickle, json, ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from json_repair import repair_json
import re
from typing import Dict, Any, Optional, Tuple

#### THIS IS FOR NEGATION JSON FIXATION
def extract_index(path):
    return int(re.search(r"qwen3_(\d+)\.pkl$", path).group(1))
def extract_llama_index(path):
    return int(re.search(r"llama31_(\d+)\.pkl$", path).group(1))

# anchor keywords (case-insensitive)
EVID_K = re.compile(
    r'''
    (["']?)          # optional opening quote
    evidence
    (["']?)          # optional closing quote
    \s*[:=]          # must be followed by : or =
    ''',
    re.I | re.X
)
CORR_K = re.compile(
    r'''
    (["']?)          # optional opening quote
    correct
    (["']?)          # optional closing quote
    \s*[:=]          # must be followed by : or =
    ''',
    re.I | re.X
)
TYPE_K = re.compile(
    r'''
    (["']?)          # optional opening quote
    type
    (["']?)          # optional closing quote
    \s*[:=]          # must be followed by : or =
    ''',
    re.I | re.X
)
BOOL_K = re.compile(
    r'''
    \b
    (true|false)
    \b
    ''',
    re.I | re.X
)

# find phenotypes key (single or double quotes)
PHEN_K = re.compile(r'["\']phenotypes["\']\s*:\s*\{', re.I)

DEMO_K = re.compile(r'["\']demographics["\']\s*:\s*\{', re.I)


def _find_demographics_block(text: str) -> Optional[str]: 
    m = DEMO_K.search(text) 
    if not m: 
        return None 
    return text[m.end():]

def reconstruct_demographics_by_anchors(text: str):
    """
    Recovery-first demographics parser.

    Supports:
      - any subset of {age, sex, ethnicity}
      - object-valued fields:
          "age": { "evidence": "...", "correct": "true" }
      - string-valued fields:
          "age": "23 years old"
    """
    block = _find_demographics_block(text)
    if block is None:
        return {}

    out = {}
    pos = 0
    n = len(block)

    FIELD_SET = {"age", "sex", "ethnicity"}

    while pos < n:
        # ---- 1) Try OBJECT-valued field ----
        m_obj = re.search(
            r'(["\'])(?P<field>age|sex|ethnicity)\1\s*[:=]\s*\{',
            block[pos:],
            re.I
        )

        # ---- 2) Try STRING-valued field ----
        m_str = re.search(
            r'(["\'])(?P<field>age|sex|ethnicity)\1\s*[:=]\s*(["\'])(?P<val>.*?)\3',
            block[pos:],
            re.I | re.S
        )

        # pick earliest valid match
        candidates = []
        if m_obj:
            candidates.append((m_obj.start(), "obj", m_obj))
        if m_str:
            candidates.append((m_str.start(), "str", m_str))

        if not candidates:
            break

        _, kind, m = min(candidates, key=lambda x: x[0])
        field = m.group("field").lower()

        # =========================
        # STRING-valued demographics
        # =========================
        if kind == "str":
            out[field] = {
                "evidence": _strip_junk_edges(m.group("val")),
                "correct": True,  # implicit correctness
            }
            pos += m.end()
            continue

        # =========================
        # OBJECT-valued demographics
        # =========================
        field_start = pos + m.end()

        em = _find_next(EVID_K, block, field_start)
        cm = _find_next(CORR_K, block, field_start)

        # --- evidence ---
        if em and cm:
            evidence_raw = _extract_span(block, em.end(), cm.start())
        elif em:
            # stop at next demographic field or end
            next_field = re.search(
                r'(["\'])(?:age|sex|ethnicity)\1\s*[:=]',
                block[field_start:],
                re.I
            )
            stop = field_start + next_field.start() if next_field else n
            evidence_raw = _extract_span(block, em.end(), stop)
        else:
            evidence_raw = ""

        # --- correct ---
        if cm:
            close_brace = block.find("}", cm.end())
            stop = close_brace if close_brace != -1 else n
            correct_raw = _extract_span(block, cm.end(), stop)
        else:
            correct_raw = ""

        out[field] = {
            "evidence": _strip_junk_edges(evidence_raw),
            "correct": _to_bool(correct_raw) if correct_raw else True,
        }

        pos = field_start

    return out


def _find_phenotype_start_from_demographics(text: str) -> Optional[int]:
    """
    Uses reconstruct_demographics_by_anchors as the oracle.
    Finds the last demographic field, then extends to its correct true/false.
    Returns absolute index where phenotypes should start.
    """
    demo = reconstruct_demographics_by_anchors(text)
    if not demo:
        return None

    # 1) find which demographic field appears LAST in text
    last_field = None
    last_field_pos = -1

    for field in demo.keys():
        # match "sex": or 'sex':
        pat = re.compile(rf'(["\']){re.escape(field)}\1\s*[:=]', re.I)
        for m in pat.finditer(text):
            if m.start() > last_field_pos:
                last_field = field
                last_field_pos = m.start()

    if last_field is None:
        return None

    # 2) from that field, find next 'correct'
    cm = CORR_K.search(text, last_field_pos)
    if not cm:
        return None

    # 3) from correct, find next true/false
    bm = BOOL_K.search(text, cm.end())
    if not bm:
        return None

    end = bm.end()

    # 4) skip trailing junk
    while end < len(text) and text[end] in ' "\'\t\r\n,}':
        end += 1

    return end

def _find_phenotypes_block(text: str) -> Optional[str]:
    # 1) if phenotypes keyword exists → original behavior
    m = PHEN_K.search(text)
    if m:
        return text[m.end():]

    # 2) fallback: derive boundary from demographics oracle
    start = _find_phenotype_start_from_demographics(text)
    if start is None:
        return None

    return text[start-1:]


def _strip_junk_edges(s: str) -> str:
    """
    Strip common separators/punctuation that surround values in broken JSON-ish text.
    Keeps internal quotes untouched; only trims edges.
    """
    if s is None:
        return s
    s = s.strip()

    # remove leading separators like : , { [
    s = re.sub(r'^[\s,:=\{\[\(]+', '', s)
    # remove trailing separators like , } ] )
    s = re.sub(r'[\s,;\}\]\)]+$', '', s)

    # strip wrapping quotes repeatedly
    s = s.strip()
    while len(s) >= 2 and (s[0] == s[-1]) and s[0] in ('"', "'"):
        s = s[1:-1].strip()

    # strip lingering edge quotes
    s = s.strip().strip('"').strip("'").strip()
    return s


def _extract_span(s: str, a: int, b: int) -> str:
    a = max(0, a)
    b = min(len(s), b)
    return s[a:b] if a < b else ""


def _find_next(pat: re.Pattern, s: str, start: int, end: int = None) -> Optional[re.Match]:
    if end:
        return pat.search(s, start, end)
    return pat.search(s, start)


def _backtrack_entity(block: str, anchor_pos: int) -> Optional[str]:
    """
    Recover entity name by looking backward from an anchor (usually evidence).
    Heuristic: closest quoted string before the anchor that looks like an entity key.
    """
    left = max(0, anchor_pos - 6000)
    window = block[left:anchor_pos]

    candidates = list(
        re.finditer(
            r'(["\'])(?P<ent>[^}\n]{1,250}?)(?<!\\)\1\s*[:=\{]',
            window,
            re.S
        )
    )

    if not candidates:
        candidates = list(
            re.finditer(
                r'(["\'])(?P<ent>[^}\n]{1,250}?)(?<!\\)\1',
                window,
                re.S
            )
        )
        if not candidates:
            return None

    ent = _strip_junk_edges(candidates[-1].group("ent"))

    if ent.lower() in {"evidence", "correct", "type", "phenotypes", "demographics"}:
        return None

    return ent


def _find_group_start(block: str, pos: int) -> int:
    """
    Minor fix you described:
    groups typically start right after a closing brace '}' (often after newline),
    followed by optional comma/whitespace, then a quote starting the next entity key.

    This finds that quote position and returns it; if not found, returns pos.
    """
    # look a bit ahead to find the first quote that begins the next key
    m = re.search(r'\{\s*,?\s*["\']', block[pos:pos + 2000])
    if m:
        # return index of the quote char
        rel = m.start()
        # find the quote within the matched snippet
        snippet = block[pos + rel: pos + rel + m.end() - m.start()]
        qrel = snippet.find('"')
        if qrel == -1:
            qrel = snippet.find("'")
        if qrel != -1:
            return pos + rel + qrel
    return pos


def _recover_evidence(block: str, group_start: int, correct_pos: int) -> Optional[str]:
    """
    Evidence missing case (as you specified):
      entity = everything from the first word of the group (right after '}' + newline + quote)
               up to before 'correct'
      evidence = "" (handled in main)
    """
    if correct_pos <= group_start:
        return ''

    # grab from group_start to correct_pos
    candidate = block[group_start:correct_pos]
    candidate = _strip_junk_edges(candidate)

    # remove any trailing separators/braces
    candidate = re.sub(r'[:=\{\[,]+$', '', candidate).strip()

    if not candidate or candidate.lower() in {"evidence", "correct", "type"}:
        return ''
    return candidate

def _is_complete_phenotype_structural(
    *,
    saw_correct: bool,
    saw_closing_brace: bool
) -> bool:
    """
    Decide whether a phenotype group is structurally complete.
    Handles truncation before 'correct' or 'type'.
    """
    # If correct is missing entirely → truncated
    if not saw_correct:
        return False

    # If no type, require a proper closing brace
    if saw_closing_brace:
        return True

    return False

def _has_future_group(block: str, pos: int) -> bool:
    """
    Check whether there is another plausible phenotype group
    (evidence or correct anchor) after position `pos`.
    """
    em = EVID_K.search(block, pos + 1)
    cm = CORR_K.search(block, pos + 1)
    return em is not None or cm is not None
def _to_bool(s: str):
    s2 = _strip_junk_edges(s).lower()
    if s2 == "true":
        return True
    if s2 == "false":
        return False
    return _strip_junk_edges(s)  # fallback if weird
def first_key_after_object(block: str, obj_key: str, start: int = 0) -> tuple[str | None, int | None]:
    """
    Find the first key inside:  "<obj_key>": { "<FIRST_KEY>": ...
    Returns (key, quote_position_of_key) or (None, None).
    """
    # Find: "phenotypes" : {
    m = re.search(rf'(["\']){re.escape(obj_key)}\1\s*:\s*\{{', block[start:], flags=re.I)
    if not m:
        return None, None

    i = start + m.end()  # position right after the '{'
    n = len(block)

    # skip whitespace
    while i < n and block[i].isspace():
        i += 1

    # next char must be quote for a key
    if i >= n or block[i] not in "\"'":
        return None, None

    q = block[i]
    j = i + 1
    while j < n:
        if block[j] == q and block[j - 1] != "\\":  # naive escape handling
            key = block[i + 1 : j]
            return key, i
        j += 1

    return None, None
def extract_string_value_for_key(block: str, key: str, key_quote_pos: int) -> str | None:
    """
    Given position of the opening quote of a key, extract its string value if it is:
      "key": "...."
    Returns the unquoted string (still raw-ish) or None if not a string value.
    """
    n = len(block)
    i = key_quote_pos

    # find colon after key
    colon = block.find(":", i)
    if colon == -1:
        return ''

    # skip whitespace after colon
    j = colon + 1
    while j < n and block[j].isspace():
        j += 1

    # must start with quote for a string value
    if j >= n or block[j] not in "\"'":
        return ''

    q = block[j]
    k = j + 1
    while k < n:
        if block[k] == q and block[k - 1] != "\\":
            return block[j + 1 : k]
        k += 1
    return ''

def reconstruct_phenotypes_by_anchors(text) -> Dict[str, Any]:
    """
    Recovery-first phenotypes parser using anchor tokens evidence / correct / type.
    Does NOT require valid JSON.

    Rules:
    - Group boundary = earliest of evidence-key or correct-key
    - If evidence is missing:
        * entity = text from group start to before 'correct'
        * evidence = ""
    - Truncation-aware:
        * If a group is incomplete AND no future group exists → stop
        * If incomplete BUT future groups exist → skip and continue
    """
    block = _find_phenotypes_block(text)
    if "phenotypes" not in block:
        block = '"phenotypes": {' + block
    if block is None:
        raise ValueError

    out: Dict[str, Any] = {}
    pos = 0
    n = len(block)

    while pos < n:
        # --- find next group anchor ---
        em = _find_next(EVID_K, block, pos)
        cm = _find_next(CORR_K, block, pos)

        anchors = [(em, "evidence"), (cm, "correct")]
        anchors = [(m, k) for m, k in anchors if m]
        if not anchors:
            break
        anchor, kind = min(anchors, key=lambda x: x[0].start())
        group_anchor_pos = anchor.start()

        # --- find group-level correct / type ---
        cm = _find_next(CORR_K, block, group_anchor_pos)
        if cm:
            tm =  _find_next(TYPE_K, block, cm.end(), cm.end()+20)
        else:
            cm2 = _find_next(CORR_K, block, group_anchor_pos)
            em2 = _find_next(EVID_K, block, em.end())
            if em2:
                tm =  _find_next(TYPE_K, block, group_anchor_pos, em2.start())
            elif cm2:
                tm =  _find_next(TYPE_K, block, group_anchor_pos, cm2.start())
            else:
                tm = None

        # =======================
        # ENTITY
        # =======================
        evidence_raw = ''
        if kind == "evidence":
            entity = _backtrack_entity(block, group_anchor_pos)
        else:
            group_start = _find_group_start(block, pos)
            evidence_raw = _recover_evidence(
                block,
                group_start,
                cm.start() if cm else group_anchor_pos
            )
            entity = _backtrack_entity(block, group_start)
            if not entity and pos == 0:
                entity, qpos = first_key_after_object(text, "phenotypes")
                evidence_raw = extract_string_value_for_key(text, entity, qpos) if entity else ''
        if not entity:
                entity = f"_unknown_{group_anchor_pos}"
        # =======================
        # EVIDENCE
        # =======================
        if len(evidence_raw) == 0:
            if kind == "evidence":
                if cm:
                    evidence_raw = _extract_span(block, em.end(), cm.start())
                elif tm:
                    evidence_raw = _extract_span(block, em.end(), tm.start())
                else:
                    next_em = _find_next(EVID_K, block, em.end())
                    stops = []
                    if next_em:
                        next_em2 = _find_next(EVID_K, block, next_em.end())
                        anchor, kind = min(anchors, key=lambda x: x[0].start())
                        if next_em2:
                            stops = [m.start() for m in (next_em, anchor) if m]
                        else:
                            # find next closing brace
                            next_brace_pos = block.find("}", em.end())
                            # collect valid stop candidates
                            if next_brace_pos != -1:
                                stops.append(next_brace_pos)                    
                    stop = min(stops) if stops else n
                    evidence_raw = _extract_span(block, em.end(), stop)
        # =======================
        # CORRECT
        # =======================
        close_brace = -1
        if cm:
            correct_raw = _extract_span(block, cm.end(), cm.end()+9)
            close_brace = block.find("}", cm.end())
        else:
            # type is missing → correct value should end at the phenotype group's closing brace `}`
            close_brace = block.find("}", em.end())
            if close_brace != -1:
                stop = close_brace
            else:
                # fallback: stop at next anchor if brace can't be found
                next_em = _find_next(EVID_K, block, em.end())
                next_em2 = _find_next(EVID_K, block, em.end())
                stops = [m.start() for m in (next_em, next_em2) if m]
                stop = min(stops) if stops else n
            correct_raw = _extract_span(block, em.end(), stop)
        # =======================
        # TYPE
        # =======================
        if tm:
            after_type = tm.end()
            close_brace = block.find("}", after_type)

            next_em = _find_next(EVID_K, block, after_type)
            next_cm2 = _find_next(CORR_K, block, after_type)
            stops = [m.start() for m in (next_em, next_cm2) if m]
            next_anchor = min(stops) if stops else n

            stop = n
            if close_brace != -1:
                stop = min(stop, close_brace)
            stop = min(stop, next_anchor)

            type_raw = _extract_span(block, after_type, stop)
            next_pos = stop
        else:
            type_raw = ""
            next_em = _find_next(EVID_K, block, group_anchor_pos + 1)
            next_cm2 = _find_next(CORR_K, block, group_anchor_pos + 1)
            stops = [m.start() for m in (next_em, next_cm2) if m]
            next_pos = min(stops) if stops else n
        # Determine group end safely
        group_end_candidates = []

        if tm:
            # if type exists, group ends at closing brace after type
            next_em = _find_next(EVID_K, block, group_anchor_pos + 1)
            next_cm = _find_next(CORR_K, block, group_anchor_pos + 1)
            end_bracket = tm.end()
            if next_em and next_cm:
                if next_em.start() < tm.start():
                    if next_em.start() < next_cm.start():
                        end_bracket = next_em.end()
                    else:
                        if next_cm.start() < tm.start():
                            end_bracket = next_cm.end()
            elif next_em:
                if next_em.start() < tm.start():
                    end_bracket = next_em.end()
            elif next_cm:
                if next_cm.start() < tm.start():
                    end_bracket = next_cm.end()
            cb = block.find("}", end_bracket)
            if cb != -1:
                group_end_candidates.append(cb + 1)
        else:
            # no type → group ends at first closing brace after correct
            if cm:
                cb = block.find("}", cm.end())
                if cb != -1:
                    group_end_candidates.append(cb + 1)
            else:
                eb = block.find("}", em.end())
                if eb != -1:
                    group_end_candidates.append(eb + 1)

        # fallback: next anchor
        group_end_candidates.append(next_pos)

        group_end = max(group_end_candidates)
        # =======================
        # STRUCTURAL COMPLETENESS
        # =======================
        saw_correct = cm is not None
        saw_type = tm is not None
        saw_closing_brace = close_brace != -1 #if tm else False
        if "true" in correct_raw.lower():
            correct_raw = 'True'
            saw_correct = True
        elif 'false' in correct_raw.lower():
            correct_raw = 'False'
            saw_correct = True
        entry = {
            "evidence": _strip_junk_edges(evidence_raw),
            "correct": _to_bool(correct_raw),
        }
        if type_raw:
            entry["type"] = _strip_junk_edges(type_raw)
        if _is_complete_phenotype_structural(
            saw_correct=saw_correct,
            saw_closing_brace=saw_closing_brace
        ):
            out[entity] = entry
            #pos = max(next_pos, group_anchor_pos + 1)
            pos = group_end
        else:
            # incomplete group → decide skip vs stop
            if _has_future_group(block, group_anchor_pos):
                #pos = group_anchor_pos + 1
                pos = group_end
                continue
            else:
                break
    return out
def extract_json_object(text):
    if "demographics" in text and "phenotypes" in text:
        try:
            return json.loads(text)
        except:
            try:
                return ast.literal_eval(text)
            except:
                pass
    demographics = reconstruct_demographics_by_anchors(text)
    phenotypes = reconstruct_phenotypes_by_anchors(text)
    final_dict = {'demographics': demographics, 'phenotypes': phenotypes}
    return final_dict

#### THIS IS FOR OUTPUT JSON FIXATION
# --- new anchors ---
HPO_K = re.compile(
    r'''
    (["']?)\s*HPO_ID\s*(["']?)   # key
    \s*[:=]\s*
    .*?                          # allow junk (non-greedy)
    (?P<hp>HP:\d{7})             # capture first valid HP id
    ''',
    re.I | re.X | re.S
)


ONSET_K = re.compile(
    r'''
    (["']?)\s*onset\s*(["']?)   # key
    \s*[:=]                     # separator
    ''',
    re.I | re.X
)

# blocks
PHEN2_K = re.compile(r'["\']phenotypes["\']\s*:\s*\{', re.I)
DEMO2_K = re.compile(r'["\']demographics["\']\s*:\s*\{', re.I)

def _find_demographics_block2(text: str) -> Optional[str]:
    m = DEMO2_K.search(text)
    return None if not m else text[m.end():]

def _find_phenotypes_block2(text: str) -> Optional[str]:
    m = PHEN2_K.search(text)
    return None if not m else text[m.end():]
def _read_scalar_value(s: str, start: int) -> Tuple[str, int]:
    """
    Read a scalar JSON-ish value starting at `start`.
    Returns (value_string, next_index).
    Handles:
      - "quoted strings" or 'quoted strings'
      - bare tokens (unknown, true, false, null, etc.)
    Stops at comma or closing brace at same nesting level for scalars.
    """
    i = start
    n = len(s)

    # skip spaces and common separators
    while i < n and s[i] in " \t\r\n:=":
        i += 1

    if i >= n:
        return "", i

    # quoted string
    if s[i] in ('"', "'"):
        q = s[i]
        i += 1
        j = i
        while j < n:
            if s[j] == q and s[j-1] != '\\':
                break
            j += 1
        val = s[i:j]
        return val, (j + 1 if j < n else n)

    # bare token
    j = i
    while j < n and s[j] not in ",}]\r\n":
        j += 1
    val = s[i:j].strip()
    return val, j

def _extract_between(block: str, start: int, end: int) -> str:
    if start < 0:
        start = 0
    if end < 0 or end > len(block):
        end = len(block)
    return block[start:end]
def _normalize_hpo_id(s: str) -> Optional[str]:
    """
    Extract exactly HP: followed by 7 digits.
    """
    if not s:
        return None
    m = re.search(r'HP:\d{7}', s)
    return m.group(0) if m else None

def reconstruct_demographics_simple(text: str) -> Dict[str, str]:
    block = _find_demographics_block2(text)
    if block is None:
        return {}

    out: Dict[str, str] = {}
    pos = 0
    n = len(block)

    # only these fields, like your earlier logic
    key_pat = re.compile(r'["\'](?P<field>age|sex|ethnicity)["\']\s*[:=]', re.I)

    while pos < n:
        m = key_pat.search(block, pos)
        if not m:
            break

        field = m.group("field").lower()
        val, next_i = _read_scalar_value(block, m.end())
        out[field] = _strip_junk_edges(val)

        pos = next_i

    return out
def reconstruct_phenotypes_hpo_onset(text: str) -> Dict[str, Dict[str, str]]:
    block = _find_phenotypes_block2(text)
    if block is None:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    pos = 0
    n = len(block)

    while pos < n:
        hm = HPO_K.search(block, pos)
        if not hm:
            break

        hpo_id = hm.group("hp")

        anchor_pos = hm.start()

        # phenotype name: reuse your backtracker
        entity = _backtrack_entity(block, anchor_pos)
        if not entity or entity.lower() in {"hpo_id", "onset", "phenotypes", "demographics"}:
            entity = f"_unknown_{anchor_pos}"
       
        # onset: search onset AFTER this HPO_ID within a reasonable window
        om = ONSET_K.search(block, hm.end())
        onset_val = "unknown"
        onset_end = hm.end()

        if om:
            onset_val, onset_end = _read_scalar_value(block, om.end())
            onset_val = _strip_junk_edges(onset_val)

        # group end: prefer closing brace after onset (or after hpo_id if onset missing)
        # this prevents re-parsing and phantom _unknown_ spam
        close_brace = block.find("}", onset_end if om else hm.end())
        if close_brace != -1:
            pos = close_brace + 1
        else:
            # fallback: advance to next HPO anchor
            next_hm = HPO_K.search(block, hm.end())
            pos = next_hm.start() if next_hm else n

        out[entity] = {"HPO_ID": hpo_id, "onset": onset_val}

    # optional: drop unknowns if you don't want them
    # out = {k:v for k,v in out.items() if not k.startswith("_unknown_")}

    return out
def valid_json(text: str) -> Dict[str, Any]:
    # common: text wrapped in outer single quotes
    if isinstance(text, str):
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            # only strip if it looks like a wrapper around an object
            if text[1:2] == "{":
                text = text[1:-1]

    # 1) strict JSON
    try:
        return json.loads(text), True # dict, complete
    except Exception:
        pass

    # 2) python literal dict
    try:
        return ast.literal_eval(text), True
    except Exception:
        pass

    # 3) repair_json (often works when trailing commas etc.)
    try:
        repaired = repair_json(text)
        return json.loads(repaired), False
    except Exception:
        pass

    # 4) anchor reconstruction
    demographics = reconstruct_demographics_simple(text)
    phenotypes = reconstruct_phenotypes_hpo_onset(text)
    return {"demographics": demographics, "phenotypes": phenotypes}, False
