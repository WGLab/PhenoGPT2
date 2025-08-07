import spacy, medspacy
from medspacy.context import ConText, ConTextRule
from medspacy.ner import TargetRule
from spacy.tokens import Doc
from ast import literal_eval
from scripts.utils import generate_output
from scripts.formatting_results import *
nlp = spacy.blank("en")
def set_target_rules( phenotype_list):
    return [TargetRule(phen, "PHENOTYPE") for phen in phenotype_list]
def include_ent(ent):
    if ent._.is_negated:
        return False
    if ent._.is_uncertain:
        return False
#         if ent._.is_historical:
#             return False
    if ent._.is_hypothetical:
        return False
    if ent._.is_family:
        return False
    return True
def remove_negation_by_medspacy(paragraph, phenotypes):
    nlp = medspacy.load()
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    texts = paragraph.split(".")
    texts = [t.strip() for t in texts if len(t) > 0]
    target_rules = set_target_rules(phenotypes)
    target_matcher.add(target_rules)
    docs = list(nlp.pipe(texts))
    positive_phens = []
    for doc in docs:
        for ent in doc.ents:
            if include_ent(ent) and str(ent) not in positive_phens:
                positive_phens.append(str(ent))
    return(positive_phens)

def remove_negation(model, tokenizer, text, phenotypes, device):
    positive_phenotypes = remove_negation_by_medspacy(text.lower(), phenotypes)
    #print(positive_phenotypes)
    data_point = {'text':text.lower(), 'phenotypes':positive_phenotypes}
    try:
        negation_response = "{'" + generate_output(model, tokenizer, data_point, temperature = 0.4, negation_detect = True, max_new_tokens = 2000, device = device)
        parsed_response = validate_json(negation_response)
        positive_phenotypes = [x for x,y in parsed_response.items() if x.lower() in text.lower() and y]
    except Exception as e:
        print(f'Cannot use LLM to detect negation: {str(e)}', flush = True)
        pass
    return positive_phenotypes