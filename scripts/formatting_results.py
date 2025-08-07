import ast
import re
import json
from collections import OrderedDict
from scripts.utils import *

def extract_first_complete_dict(text):
    start = text.find('{')
    if start == -1:
        return None
    stack = []
    for i in range(start, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if not stack:
                return text[start:i+1]
    return None  # No complete top-level dict found

def normalize_keys(obj):
    if isinstance(obj, dict):
        normalized = {}
        for k, v in obj.items():
            key = "HPO_ID" if k.lower() in {"hpo", "h_id", "hpo_id"} else k
            normalized[key] = normalize_keys(v)
        return normalized
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    return obj

def normalize_hpo_ids(phenotypes):
    cleaned = OrderedDict()
    try:
        for term, data in phenotypes.items():
            if isinstance(data, str):
                hpo_id = data
                onset = "unknown"
            elif isinstance(data, dict):
                normalized = {k.lower(): v for k, v in data.items()}
                hpo_id = normalized.get("hpo_id") or normalized.get("hpo") or normalized.get("h_id") or impute_missing_hpo(term)
                onset = normalized.get("onset", "unknown")
            else:
                hpo_id = impute_missing_hpo(term)
                onset = "unknown"

            # Fix malformed or missing HPO IDs
            if not isinstance(hpo_id, str) or not re.match(r"^HP:\d{7}$", hpo_id):
                hpo_id = impute_missing_hpo(term)

            cleaned[term] = {
                "HPO_ID": hpo_id,
                "onset": onset
            }
    except Exception as e:
        print()
        print(f"Error when normalizing HPO ID field: {str(e)}", flush=True)
        print(response, flush=True)
        print()
        cleaned = {}
    return cleaned
def validate_json(raw_text):
    raw_response = extract_first_complete_dict(raw_text)
    try:
         # Safely parse the string into a Python dictionary
        data = ast.literal_eval(raw_response)

        # Convert string "True"/"False" to actual booleans
        cleaned = {
            key: (value if isinstance(value, bool) else str(value).strip().lower() == 'true')
            for key, value in data.items()
        }

        # Return as JSON
        return cleaned
    except Exception as e:
        if "'{' was never closed" in str(e):
            raw_response = raw_response + "}"
            try:
                # Safely parse the string into a Python dictionary
                data = ast.literal_eval(raw_response)

                # Convert string "True"/"False" to actual booleans
                cleaned = {
                    key: (value if isinstance(value, bool) else str(value).strip().lower() == 'true')
                    for key, value in data.items()
                }

                # Return as JSON
                return cleaned
            except:
                pass
    try:
        # Safely parse the string into a Python dictionary
        data = json.loads(raw_response)

        # Convert string "True"/"False" to actual booleans
        cleaned = {
            key: (value if isinstance(value, bool) else str(value).strip().lower() == 'true')
            for key, value in data.items()
        }

        # Return as JSON
        return cleaned
    except:
        return {}
def fix_and_parse_json(raw_text):
    block = extract_first_complete_dict(raw_text)
    if not block:
        return {"demographics": {}, "phenotypes": {}}
    
    parsed = None

    # Try ast.literal_eval first
    try:
        parsed = ast.literal_eval(block)
    except Exception:
        # Fallback to JSON-style parse
        try:
            block = block.replace("'", '"')
            block = re.sub(r',(\s*})', r'\1', block)
            parsed = json.loads(block)
        except Exception:
            return {"demographics": {}, "phenotypes": {}}

    parsed = normalize_keys(parsed)
    demographics = parsed.get("demographics", {})
    phenotypes = normalize_hpo_ids(parsed.get("phenotypes", {}))

    return {
        "demographics": demographics,
        "phenotypes": dict(phenotypes)
    }
