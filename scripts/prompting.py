def prompt_negation(pheno_dict):
    CLINICAL_NOTE = pheno_dict['clinical_note']
    AGE = pheno_dict['demographics']['age']
    SEX = pheno_dict['demographics']['sex']
    ETHNICITY = pheno_dict['demographics']['ethnicity']
    PHENOTYPE_LIST = list(pheno_dict['phenotypes'].keys())
    instruction = f"You are a clinical information verification assistant. Your task is to verify whether the extracted demographics and phenotypes are correctly supported by the clinical note."
    question = f"""You MUST return a single valid JSON object and nothing else.

=====================
CRITICAL OUTPUT RULES
=====================
1) Output MUST be valid JSON:
   - Use double quotes for all keys and strings.
   - Use JSON booleans: true / false (NOT "True"/"False").
   - Use null for missing evidence.
   - Do NOT include explanations, comments, or extra text.

2) Required keys:
   - Top-level keys must be exactly: "demographics", "phenotypes".
   - For EACH phenotype key in EXTRACTED_DATA, output an object with EXACTLY three fields:
       "evidence", "correct", "type"
     NEVER omit "type".

3) Consistency rule (NO EXCEPTIONS):
   - If type == "patient" OR type == "patient_indirect" → correct MUST be true
   - If type != "patient" AND type != "patient_indirect" → correct MUST be false

4) Evidence rules:
   - Evidence must be a short verbatim phrase or sentence (less than 20 words) copied from the clinical note, ideally 'containing that extracted phenotypes'.
   - Evidence used to support a phenotype MUST explicitly describe the PATIENT.
   - Evidence must NOT be identical to the phenotype key alone.
   - You may use three dots ("…") to shorten evidence, but must preserve defining values and descriptors.
   - If no relevant text exists, set evidence = null.
   - Do NOT invent or hallucinate evidence.
5) DISEASE AND SYNDROME TERMS (IMPORTANT)

Disease or syndrome names MAY represent valid phenotypes when they describe
a pathological condition affecting the patient (e.g., cancer, neurocutaneous disease,
neurodevelopmental disorder, structural disease).

Examples of VALID phenotypes:
- Neurofibromatosis
- Basal cell carcinoma
- Cardiomyopathy
- Hyperparathyroidism
- Epilepsy

If the term refers to a pathological disease state affecting the patient, classify as:
  type = "patient"
  correct = true

However, genetic diagnostic labels that only identify a named genetic condition
without describing a pathological clinical abnormality should NOT be treated
as phenotypes such as: Angelman syndrome, 22q11 deletion syndrome.

If the term is used only as a genetic diagnosis label or administrative diagnosis without describing the abnormal clinical state itself, classify as:
  type = "reference"
  correct = false

=====================
MANDATORY DECISION ORDER (VERY IMPORTANT)
=====================
You MUST follow this order for EACH phenotype:

STEP 1: Locate relevant evidence in the note (direct OR indirect). The EVIDENCE is the deciding factor.
STEP 2: Check for NEGATION-AS-NORMALITY.
STEP 3: Check for ABSENCE-AS-PATHOLOGY.
STEP 4: Check for INDIRECT / DEFINITION-BASED CONFIRMATION.
STEP 5: Only AFTER the above, decide lab / non_phenotype / uncertainty.

Skipping or reordering these steps is NOT allowed.

=====================
PHENOTYPE DEFINITIONS
=====================
A medical phenotype is an abnormal sign, symptom, exam finding,
or clinically meaningful abnormal lab or imaging finding describing the PATIENT.

=====================
NEGATION VS ABSENCE (HPO-AWARE)
=====================
Words like "no", "absent", "without", or "lack" do NOT always indicate negation.

There are TWO different concepts:

1) ABSENCE-AS-PATHOLOGY (POSITIVE PHENOTYPE):
   Some phenotypes explicitly describe absence, agenesis, or lack of a structure or function.
   These are VALID abnormal phenotypes.

   Examples (if affirmed in the note):
   - Absence of toe
   - Agenesis of corpus callosum
   - Absent patella
   - Missing teeth
   - Absent kidney
   - Lack of speech
   - Inability to sweat

   Rule:
   - If the phenotype itself encodes biological absence or non-formation,
     and the note affirms this condition,
     classify as:
       type = "patient"
       correct = true
   - Do NOT classify these as "negated".

2) NEGATION-AS-NORMALITY (TRUE NEGATION):
   Negation applies ONLY when the note states a phenotype is NOT present,
   implying normality, exclusion, or denial.

   Examples:
   - "No seizures"
   - "Denies vomiting"
   - "Normal digits"
   - "No evidence of renal agenesis"
   - "Absent dysmorphic features"

   Rule:
   - Use type = "negated" ONLY when the phenotype is explicitly denied or ruled out.

=====================
NEGATION / NORMALITY HANDLING
=====================
A "negation cue" is ANY word/phrase/symbol in the note that indicates the finding is absent, denied, excluded,
normal, resolved, absence-as-normality, or explicitly negative.

IMPORTANT:
- This is OPEN-SET. Do NOT restrict yourself to a fixed list.
- If you set type = "negated", your evidence MUST quote the exact cue phrase found in the note that causes negation.

Common cue examples (NON-EXHAUSTIVE; only examples):
- denial/absence: "denies", "no", "not", "without", "never", "none", "negative for", "free of", etc
- normality: "normal", "within normal limits", "unremarkable", "benign", "intact", etc
- exclusion/rule-out: "rule out", "ruled out", "exclude", "excluded", "not consistent with", etc
- resolution: "resolved", "improving", "no longer", etc
- shorthand/symbols: "(-)", "neg", "wnl", "NAD", "afebrile", etc

HARD RULE FOR type="negated":
- Evidence MUST contain BOTH:
  (1) phenotype context AND
  (2) at least one explicit verbatim cue phrase/symbol indicating negation/normality/exclusion/resolution.
- If you cannot quote such a cue from the note, you MUST NOT label it as "negated".
  Use type="uncertainty" (correct=false) instead.

=====================
INDIRECT / DEFINITION-BASED PHENOTYPES (SUBJECT-AWARE, MANDATORY)
=====================
Some phenotypes are NOT stated explicitly by name in the clinical note,
but are implied by measurements, scores, ranges, developmental milestones,
functional descriptions, or clinical assessments.

The phenotype key may be underspecified, abbreviated, or non-diagnostic
(e.g., "iq", "speech", "development", "height").

HOWEVER, indirect inference is ONLY valid when the evidence and the phenotype
belong to the SAME SUBJECT.

---------------------
STEP 0: SUBJECT SCOPE CHECK (HIGHEST PRIORITY — DO FIRST)
---------------------
Before ANY indirect or definition-based reasoning, you MUST determine
WHO the phenotype mention refers to based on sentence- or clause-level attribution.

RULES:
1) If the phenotype term appears in a sentence or clause that explicitly
   refers to a FAMILY MEMBER or FAMILY HISTORY
   (e.g., mentions of cousin, sibling, parent, uncle, aunt, family history),
   then the phenotype MUST be classified as:
     type = "family"
     correct = false
   This applies REGARDLESS of the phenotype name or wording.

2) You MUST NOT use evidence from a different sentence or clause
   to validate, infer, or promote a phenotype.
   Evidence is ONLY valid if it comes from the SAME sentence
   or a clearly patient-referential clause as the phenotype mention.

3) If a phenotype is mentioned ONLY in a family-attributed sentence or clause,
   it MUST remain a family phenotype EVEN IF:
   - the patient has related abnormalities elsewhere in the note, OR
   - the phenotype could be inferred from patient evidence.

4) Indirect or definition-based reasoning is STRICTLY DISALLOWED
   for any phenotype that is family-attributed.

SUMMARY:
- Subject attribution overrides phenotype semantics.
- Phenotype wording does NOT matter.
- Family attribution ALWAYS wins over inference.

---------------------
STEP 1: EVIDENCE-FIRST EVALUATION (PATIENT-ONLY)
---------------------
If and ONLY IF the phenotype belongs to the PATIENT, proceed.

The phenotype key itself does NOT determine correctness or type.
The EVIDENCE is the deciding factor.

RULES:
1) You MUST evaluate whether the quoted evidence describes a
   clinically meaningful abnormality in the PATIENT.

2) The phenotype MAY be confirmed even if:
   - the phenotype name is not explicitly stated, OR
   - the phenotype key is underspecified (e.g., "iq", "speech"), OR
   - the evidence uses measurements, scores, ranges, or descriptions.

3) If the evidence clearly satisfies accepted clinical definitions
   or standard clinical reasoning for an abnormal phenotype:
   - type MUST be "patient" or "patient_indirect"
   - correct MUST be true

4) You MUST rely on the evidence itself.
   Do NOT require the phenotype name to appear verbatim.

---------------------
STEP 2: INDIRECT / DEFINITION-BASED CONFIRMATION
---------------------
Indirect confirmation applies when evidence implies a phenotype
through clinical criteria rather than explicit naming.

VALID indirect evidence includes (non-exhaustive):
- Quantitative values or ranges (scores, percentiles, SDs)
- Developmental milestone timing
- Functional or cognitive assessments
- Qualitative clinical descriptions implying impairment

RULES:
1) The evidence MUST include the defining abnormal value, range,
   or description.
2) The inference MUST be standard in clinical practice.
3) Do NOT invent thresholds or criteria not present in the note.
4) Vague or subjective phrases without defining information
   (e.g., "low IQ", "developmental concerns") → uncertainty.

If indirect confirmation succeeds:
- type MUST be "patient_indirect"
- correct MUST be true

---------------------
STEP 3: NEGATION AND NORMALITY
---------------------
Indirect inference does NOT override negation rules.

If the evidence explicitly states normality, denial, exclusion,
or resolution of the phenotype in the PATIENT:
- apply negation logic
- type MUST be "negated"
- correct MUST be false

Absence of an explicit phenotype name is NOT negation.

---------------------
STEP 4: FAILURE CONDITIONS
---------------------
Only AFTER all steps above:

1) Assign "non_phenotype" ONLY IF:
   - the evidence does NOT describe any abnormal clinical state, AND
   - no clinically valid phenotype can be inferred.

2) Assign "lab" ONLY IF:
   - the term is a laboratory test or measurement, AND
   - the evidence does NOT indicate an abnormal phenotype.

3) If evidence is insufficient or ambiguous:
   - type = "uncertainty"
   - correct = false

---------------------
CRITICAL CONSTRAINTS (DO NOT VIOLATE)
---------------------
- You MUST NOT validate a phenotype using evidence from a different subject.
- You MUST NOT promote a family-only phenotype using patient evidence.
- You MUST NOT reject a phenotype solely because the key is underspecified.
- The EVIDENCE always overrides the surface form of the key.

=====================
LAB PHENOTYPE RULE (MANDATORY)
=====================
Laboratory findings CAN represent valid phenotypes.

RULES:
1) If a lab test shows an ABNORMAL result describing the PATIENT
   (e.g., elevated, decreased, low, high, abnormal, reduced, absent, * with abnormal lab value/range),
   it MUST be treated as a phenotype.

   → type = "lab"
   → correct = true

2) If a lab test is mentioned WITHOUT abnormality
   (e.g., test name only, pending, normal, within normal limits, numeric values without *):
   → type = "lab"
   → correct = false

3) Negated or normal lab findings (e.g., "no anemia", "normal calcium"):
   → type = "negated"
   → correct = false

IMPORTANT:
- Do NOT classify abnormal lab phenotypes as "non_phenotype".
- Do NOT require mapping to a named disease or syndrome.

=====================
EXPLICIT NON-PHENOTYPES (correct=false)
=====================
- Any family history condition, hereditary information, additional references, or other persons’ conditions.
  (e.g., “mother with lung cancer”, “father with COPD”, “sibling had seizures”)
- Any lab test name, unit, culture, biomarker, or imaging modality unless it explicitly includes 
  an ABNORMAL RESULT describing the patient’s phenotype or provides SIGNIFICANT findings.
  (Examples that are NOT phenotypes: “WBC”, “IgE pending”, “urine culture”, “MRI”, “escherichia coli”)
- Any normal or non-abnormal finding (“normal mucosa”, “normal exam”, “no edema”, “clear lungs”, "symmetric palate", "eom intact").
- Any procedural, administrative, or diagnostic terms (e.g., “TTE”, “nasopharyngeal laryngoscopy”, “s/p repair”, “discharge diagnosis section”)
- Any term that is a measurement artifact, score, noun, abbreviation, neuropsychological task, external medical test (i.e. sequencing test), or observational phrase
  that does NOT represent an abnormal sign/symptom. (Examples: “s1/2”, “regular rhythm”, “O2 sat”, “AFB pending”, “postnasal drip evaluation”, "sequencing")
- Any vague or non-medical term or phrase that does not convey an abnormal clinical finding.

=====================
TYPE OPTIONS
=====================
Allowed values for "type":
- "patient"
- "patient_indirect"
- "negated"
- "family"
- "non_phenotype"
- "lab"
- "imaging"
- "procedure"
- "reference"
- "nonspecific"
- "uncertainty"

Type selection priority:
1) Affirmed abnormal phenotype (including absence-as-pathology) → "patient"
2) Indirect definition-based confirmation → "patient_indirect"
3) Explicitly denied / normal / ruled out → "negated"
4) Family members → "family"
5) Test/procedure name without abnormality → "lab" / "imaging" / "procedure"
6) Clearly not a phenotype → "non_phenotype"
7) Vague or unclear → "uncertainty" or "nonspecific"

=====================
DEMOGRAPHICS VALIDATION
=====================
For age, sex, ethnicity:

- If extracted value == "unknown":
    correct = true ONLY if the note does NOT mention the field.
- If extracted value != "unknown":
    correct = true ONLY if the note explicitly matches the value.
- Otherwise correct = false.
- Empty placeholders (empty or ___) are NOT valid values.

=====================
REQUIRED OUTPUT FORMAT
=====================
{{
  "demographics": {{
    "age": {{"evidence": "<short phrase supporting or refuting>", "correct": True/False}},
    "sex": {{"evidence": "<short phrase supporting or refuting>", "correct": True/False}},
    "ethnicity": {{"evidence": "<short phrase supporting or refuting>", "correct": True/False}}
  }},
  "phenotypes": {{
    "<verbatim phenotype_1>": {{"evidence": "<short phrase supporting or refuting>", "correct": True/False, "type": "<whether it is patient, negated, family, non_phenotype, lab, etc>"}},
    "<verbatim phenotype_2>": {{"evidence": "<short phrase supporting or refuting>", "correct": True/False, "type": "<whether it is patient, negated, family, non_phenotype, lab, etc>"}}
  }}
}}

IMPORTANT:
- Phenotype keys MUST exactly match the extracted phenotype strings.
- Include EVERY extracted phenotype exactly once.
- Do NOT add or remove phenotype keys.

=====================
INPUT
=====================
CLINICAL_NOTE:
<<<
{CLINICAL_NOTE.replace("'","").replace('"','')}
>>>

EXTRACTED_DATA:
<<<
Demographics:
- age: {AGE}
- sex: {SEX}
- ethnicity: {ETHNICITY}

Phenotypes:
{PHENOTYPE_LIST}
>>>

=====================
TASK
=====================
Validate each demographic field and each phenotype using the rules above.

FINAL SILENT CHECK BEFORE OUTPUT:
- Valid JSON only?
- Every phenotype has evidence, correct, and type?
- Indirect phenotypes checked BEFORE lab/non_phenotype?
- IQ / milestones NOT labeled as lab?
- Negation vs absence handled correctly?

Return ONLY the JSON object."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question}
      ]
    return messages
def generate_prompt(data_point):
    instruction = (
        "You are a genetic counselor specializing in extracting demographic details "
        "and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. "
        "Your task is to provide accurate and concise information without generating random answers. "
        "When demographic details or phenotype information is not explicitly mentioned in the input, "
        "use 'unknown' as the value."
    )

    question = (
        "Read the following input text and generate a JSON-formatted output with the following keys: "
        "demographics and phenotypes. For the demographics key, create a sub-dictionary with age, sex, "
        "and ethnicity as keys. For the phenotype key, create a sub-dictionary where each HPO term is a key, "
        "and the value is a sub-dictionary that contains corresponding HPO identifier (HPO_ID) and the "
        "patient’s age (onset) when the phenotype first appeared, if mentioned in the text. "
        "If any information is unavailable, return 'unknown' for that field.\nInput: "
    )
    messages = [
      {"role": "system", "content": instruction},
      {"role": "user", "content": question + data_point.replace("'","").replace('"','')}
    ]
    return messages
