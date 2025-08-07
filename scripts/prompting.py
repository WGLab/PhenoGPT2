def prompt_negation(data_point):
    instruction = "You are a genetic counselor specializing in classifying whether given terminologies are actual medical phenotypes and whether they are present or absent based on the provided text."
    #question = """Analyze the input text and the list of phenotypes provided. Generate a JSON object where each phenotype is a key, assigned True if the term represents a human phenotype ontology (abnormal physical, biochemical, and behavioral traits) and it affects patient's health.\nAssign False to a phenotype if any of the following conditions are true:\n1. The patient does not exhibit the phenotype.\n2. The patient carries only normal traits (i.e., no abnormalities).\n3. The phenotype does not apply to the patient.\n4. The term is not an actual medical phenotype, including:\n   - General or vague terms that do not describe a specific trait or abnormality.\n   - Genomic locations (e.g., chromosome positions).\n   - Treatments or therapies.\n   - Chemical compounds or drugs.\n   - Medical tests or procedures.\n\nImportant: Be cautious when interpreting negations or prefixes that may reverse the meaning (e.g., 'non-', 'un-', 'no evidence of', 'normal', 'regular', etc.). For example: 'normal intelligence' is not a phenotype so assign False.\nOutput only the JSON object without any additional text or formatting.\nInput:\n"""
    question = """Analyze the input text and the list of phenotypes provided. Generate a JSON object where each phenotype is a key. Assign the value `True` only if:
    - The term represents a human phenotype ontology (HPO) concept related to abnormal physical, biochemical, or behavioral traits (abbreviations or synonyms are accepted).
    - The phenotype clearly affects the patient's health.
    - Tagged medical codes (ICD-10, ICD-11, Snomed CT, HPO IDs).
    - Patient is noteworthy for these phenotypes.
    - Misspelled or typos are still accepted if they are truly phenotypes.
    
    Assign `False` in any of the following situations:
    1. The term does **not** describe an abnormal human phenotype (i.e., not a recognized Human Phenotype Ontology (HPO) term related to physical, biochemical, or behavioral abnormalities).
    2. The patient does **not** exhibit the phenotype.
    3. The patient exhibits only **normal** traits (i.e., no abnormalities).
    4. The phenotype belongs to other people rather than the patient (i.e., father, mother, siblings, aunt, uncle).
    5. The phenotype is mentioned from any publications or references that patient does not experience.
    6. The term is **not a valid medical phenotype**, including:
    - General or vague terms without specific clinical meaning.
    - Genetic variant names (e.g., "p.Glu123Lys", "c.456+1G>A").
    - Genomic locations or coordinates (e.g., "chr2q31").
    - Treatments, therapies, or medical interventions.
    - Drugs, chemicals, or compounds.
    - Names of medical tests or procedures.

    ⚠️ Pay special attention to **negation cues or prefixes** that reverse meaning:
    - Examples: "no evidence of", "does not have", "non-", "un-", "normal", "within normal limits", "regular", etc.
    - For example, phrases like "normal intelligence" or "no signs of hypotonia" indicate that the phenotype should be assigned `False`.

    Only output a JSON object (dictionary) without any additional text or formatting like the following example:

    Example Input:
        Text: "The patient has microcephaly and dev delay but normal muscle tone. Labs: WBC 1.0, Hgb 13.6. Cornelia de Lange Syndrome is a rare genetic disorder associated with growth retardation and distinctive facial features."
        Phenotypes: microcephaly; dev delay; normal muscle tone; WBC 1.0; Hgb 13.6; growth retardation; distinctive facial features

    Example Output:
        {
            'microcephaly': True,
            'dev delay': True,
            'normal muscle tone': False,
            'WBC 1.0': True,
            'Hgb 13.6': False,
            'growth retardation': False,
            'distinctive facial features': False
        }
    Real Input:
        Text: """
    text = data_point['text']
    if isinstance(data_point['phenotypes'], list):
        phenotypes = "; ".join(data_point['phenotypes'])
    elif isinstance(data_point['phenotypes'], str):
        pass
    else:
        phenotypes = "; ".join(list(data_point['phenotypes'].keys()))
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    {{'"""
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + text.replace("'","").replace('"','') + f"\nPhenotypes: {phenotypes}" ,
                                )

    return prompt
def generate_prompt(data_point):
#     instruction = "You are a genetic counselor specializing in extracting demographic details and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. Your task is to provide accurate and concise information without generating random answers. When demographic details or phenotype information is not explicitly mentioned in the input, use 'unknown' as the value."
#     question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes.For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and its corresponding HPO identifier is the value. If any information is unavailable, return 'unknown' for that field.\nInput:\n"
    instruction = "You are a genetic counselor specializing in extracting demographic details and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. Your task is to provide accurate and concise information without generating random answers. When demographic details or phenotype information is not explicitly mentioned in the input, use 'unknown' as the value."
    question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and the value is a sub-dictionary that contains corresponding HPO identifier ('HPO_ID') and the patient’s age ('onset') when the phenotype first appeared, if mentioned in the text. If any information is unavailable, return 'unknown' for that field.\nInput: "
    #question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and the value is a list that contains corresponding HPO identifier and followed by the patient’s age (onset) when the phenotype first appeared, if mentioned in the text. If any information is unavailable, return 'unknown' for that field.\nInput: "
    #question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics, phenotypes, and onset. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and its corresponding HPO identifier is the value. For the onset key, create a sub-dictionary where each HPO term is a key, and the patient’s age (onset) when that phenotype first appeared, if mentioned in the text is the value. If any information is unavailable, return 'unknown' for that field.\nInput:\n"

    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    |==|Response|==|
    {{'demographics': {{'age': '"""
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + data_point.replace("'","").replace('"',''),
                                )

    return prompt
def generate_prompt_alternative(data_point):
#     instruction = "You are a genetic counselor specializing in extracting demographic details and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. Your task is to provide accurate and concise information without generating random answers. When demographic details or phenotype information is not explicitly mentioned in the input, use 'unknown' as the value."
#     question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes.For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and its corresponding HPO identifier is the value. If any information is unavailable, return 'unknown' for that field.\nInput:\n"
    instruction = "You are a genetic counselor specializing in extracting demographic details and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. Your task is to provide accurate and concise information without generating random answers. When demographic details or phenotype information is not explicitly mentioned in the input, use 'unknown' as the value."
    question = """Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and the value is a sub-dictionary that contains corresponding HPO identifier (HPO_ID) and the patient’s age (onset) when the phenotype first appeared, if mentioned in the text. If any information is unavailable, return 'unknown' for that field.\n
Example Input:
"The patient is a 10-year-old Vietnamese male who presents with cleft palate and seizures. He was found with dev delay at age of 2."

Example Output:
{
  "demographics": {
    "age": "10-year-old",
    "sex": "male",
    "ethnicity": "Vietnamese",
    "race": "unknown"
  },
  "phenotypes": {
    "cleft palate": {
      "HPO_ID": "HP:0000175",
      "onset": "unknown"
    },
    "seizures": {
      "HPO_ID": "HP:0001250",
      "onset": "unknown"
    },
    "dev delay": {
      "HPO_ID": "HP:0001263",
      "onset": "age of 2"
    },
  }
}
Now, please read the input below and provide the answer. Do not include the example output in your result. Input: """
    #question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and the value is a list that contains corresponding HPO identifier and followed by the patient’s age (onset) when the phenotype first appeared, if mentioned in the text. If any information is unavailable, return 'unknown' for that field.\nInput: "
    #question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics, phenotypes, and onset. For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and its corresponding HPO identifier is the value. For the onset key, create a sub-dictionary where each HPO term is a key, and the patient’s age (onset) when that phenotype first appeared, if mentioned in the text is the value. If any information is unavailable, return 'unknown' for that field.\nInput:\n"

    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    |==|Response|==|
    {{'demographics': {{'age': '"""
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + data_point.replace("'","").replace('"',''),
                                )
    return prompt