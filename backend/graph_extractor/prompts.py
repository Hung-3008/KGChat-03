from typing import Dict

PROMPT_TEMPLATE = """You are a medical entity extraction expert. Extract all biomedical and clinical entities from the following medical text.

Instructions:
- Extract entities such as: diseases, symptoms, medications, procedures, anatomical terms, diagnostic tests, treatments, and clinical findings
- For each entity, provide:
  * name: The canonical/normalized form of the entity (e.g., "diabetes mellitus" not "DM")
  * mention: The exact context sentence or phrase from the text containing this entity
- Only extract factual entities explicitly mentioned in the text
- Return valid JSON matching the schema with an "entities" list

Text:
"""


ACTIVITY_PROMPT = '''Extract medical procedures and healthcare activities.

Semantic types:
- Health_Care_Activity: General healthcare activities
- Laboratory_Procedure: Lab tests, analysis
- Diagnostic_Procedure: Imaging, exams, tests
- Therapeutic_or_Preventive_Procedure: Surgeries, treatments, prevention
- Research_Activity: Clinical trials, protocols, molecular techniques
- Behavior: Clinically significant behaviors

Rules:
1. Extract EXACT phrases.
2. Choose MOST SPECIFIC type.
3. IGNORE: daily activities, general behaviors, connector verbs.

Text:
[INPUT TEXT]
'''

PHENOMENON_PROMPT = '''Extract diseases, symptoms, and pathologies.

Semantic types:
- Disease_or_Syndrome: Diseases, disorders, syndromes, mental/behavioral dysfunctions
- Neoplastic_Process: Cancers, tumors
- Injury_or_Poisoning: Injuries, trauma, poisoning
- Pathologic_Function: Abnormal biological/cellular/molecular functions
- Sign_or_Symptom: Observable signs, reported symptoms

Rules:
1. Extract EXACT phrases.
2. Choose MOST SPECIFIC type.
3. IGNORE: normal biological processes unless part of disease mechanism.

Text:
[INPUT TEXT]
'''

PHYSICAL_OBJECT_PROMPT = '''Extract anatomical structures and substances.

Semantic types:
- Body_Part_Organ_or_Organ_Component: Organs, body parts
- Tissue: Tissues
- Cell: Cells
- Gene_or_Genome: Genes, genetic material
- Anatomical_Abnormality: Congenital or acquired abnormalities
- Clinical_Drug: Medications, drugs
- Pharmacologic_Substance: Substances, antibiotics, hormones, enzymes, vitamins
- Medical_Device: Devices, implants
- Body_Substance: Fluids, excretions
- Organism: Pathogens, viruses, bacteria

Rules:
1. Extract EXACT phrases.
2. FOCUS on: anatomies, drugs, devices.
3. IGNORE: food, common chemicals.
4. Organism: ONLY if disease-causing.

Text:
[INPUT TEXT]'''

CONCEPTUAL_ENTITY_PROMPT = '''Extract clinical findings and measurements.

Semantic types:
- Finding: Clinical findings, attributes, observations
- Laboratory_or_Test_Result: Lab results, test outcomes
- Anatomical_Concept: Body locations, regions, systems
- Quantitative_Concept: Measurements, scores
- Temporal_Concept: Time-related info

Rules:
1. Extract EXACT phrases.
2. STRICT: Only extract if clinically relevant.
3. IGNORE: generic terms ("results", "study"), occupations.

Text:
[INPUT TEXT]'''


CONTEXT_ENTITY_FILTER_PROMPT = """Validate extracted clinical entities based on context and clinical relevance.

INPUT:
- Clinical text
- Entities grouped by semantic type

VALIDATION CRITERIA:
1. CONTEXT: Is entity a core concept (not just mentioned in passing)?
2. RELEVANCE: Explicitly relevant to this patient/case?
3. NEGATION: Free from "no evidence of", "rule out", "possible" (unless about ruling out)?
4. SEMANTIC FIT: Truly belongs to assigned type in this context?
5. CLINICAL SIGNIFICANCE: Directly relates to diagnosis/treatment/symptoms?

RULES:
- For duplicate entities at same hierarchy depth: keep most clinically relevant type
- For phrases containing other entities: keep more specific if it adds clinical context
- When uncertain: prefer entities directly related to patient care
- Extract exact mention text from original document

## CLINICAL TEXT
[CLINICAL_INPUT_TEXT]

## ENTITIES TO EVALUATE
[ENTITIES_INPUT]

## OUTPUT
Return ONLY valid JSON:
{
  "entities": [
    {
      "name": "canonical name",
      "semantic_type": "specific semantic type",
      "mention": "exact text from document"
    }
  ]
}
"""

EDGE_EXTRACTION_PROMPT = """
You are a medical expert. Extract relationships from the text using the entities below.

Rules:
- Only use information within the provided clinical text.
- The relationship type must be one of: [treats, causes, associated_with, side_effect_of, diagnosed_by].
- Only create relationships between entities that appear in the provided entity list.

For each relationship, return a JSON object with:
- subject: exact entity name from the entity list
- predicate: one of the allowed relationship types
- object: exact entity name from the entity list
- evidence: exact sentence from the text that supports the relationship

If there is no clear relationship, return {"edges": []}.

Clinical text:
[INPUT TEXT]

Provided entities (exact names):
[ENTITIES LIST]

Return a JSON object with a key "edges" containing a list of relationship objects.
"""

EDGE_VALIDATION_PROMPT = """
Here is the English translation of the prompt:

"You are a clinical medical expert with 15 years of experience, and also an auditor for a medical knowledge graph system. Your task is to VERIFY THE ACCURACY of the proposed relationships extracted from a medical text, based on 3 strict criteria:
1. DIRECT EVIDENCE (Must be an exact citation from the text)
2. CLINICAL PLAUSIBILITY (Consistent with treatment guidelines & pathogenesis)
3. CONSISTENCY (Must not contradict other information in the text)

### INPUT TO PROCESS
Medical Text:
[INPUT TEXT]

LIST OF VALID ENTITIES (Only use entities in this list):
[ENTITY_LIST]

PROPOSED RELATIONSHIPS to verify:
[PROPOSED_RELATIONSHIPS]

### MANDATORY VERIFICATION RULES
1. HALLUCINATION BLOCKER:
   - IMMEDIATELY REJECT if:
      - An Entity does not exist in the LIST OF VALID ENTITIES
      - The Relation is not in the allowed list:
        [diagnoses, detects, observes, treats, side_effect_of, associated_with, contraindicated_with, administered_for, caused_by]
      - The Evidence consists of >1 sentence or does not contain BOTH entities in the same sentence.
   - ONLY ACCEPT if there is an EXACT, WORD-FOR-WORD CITATION containing both entities and the stated relationship.

2. CLINICAL CHECK:
   - Use domain expertise to detect:
      - Drug not indicated for the disease (e.g., Lisinopril does not treat diabetes)
      - Illogical causal relationships (e.g., E. coli infection does not cause elevated HbA1c)
      - Incorrect diagnostic attribution (e.g., "family history of X" != "patient has X")

3. HANDLING AMBIGUITY:
   - If the evidence is unclear -> REJECT instead of inferring.
   - If the relation is close but imprecise -> PROPOSE A CORRECTION (e.g., Replace "causes" with "associated_with_elevated_levels").
"""

