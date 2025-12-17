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


ACTIVITY_PROMPT = '''Extract clinical activity entities from the text.

Key semantic types:
- Laboratory_Procedure: Lab tests (e.g., "blood test", "biopsy")
- Diagnostic_Procedure: Diagnostic tests (e.g., "MRI", "CT scan")
- Therapeutic_or_Preventive_Procedure: Treatments (e.g., "surgery", "chemotherapy")
- Health_Care_Activity: Healthcare activities

Rules:
1. Extract EXACT text spans only
2. Use MOST SPECIFIC semantic type available
3. FOCUS on clinical procedures and tests
4. IGNORE: general daily activities, connector verbs, non-clinical behaviors
5. Return valid JSON matching the schema

Text:
[INPUT TEXT]
'''

PHENOMENON_PROMPT = '''Extract clinical phenomenon entities from the text.

Key semantic types:
- Disease_or_Syndrome: Diseases or syndromes
- Neoplastic_Process: Cancer/tumor-related processes
- Injury_or_Poisoning: Injuries or poisonings
- Pathologic_Function: Pathological functions
- Mental_or_Behavioral_Dysfunction: Mental/behavioral dysfunctions
- Sign_or_Symptom: Clinical signs or symptoms

Rules:
1. Extract EXACT text spans only
2. Use MOST SPECIFIC semantic type
3. PRIORITIZE diseases, cancers, injuries, and pathologic functions
4. IGNORE: general biological functions unless explicitly abnormal
5. Return valid JSON matching the schema

Text:
[INPUT TEXT]
'''

PHYSICAL_OBJECT_PROMPT = '''Extract clinical physical object entities from the text.

Key semantic types:
- Body_Part_Organ_or_Organ_Component: Organs, body parts
- Tissue: Tissues
- Cell: Cells
- Gene_or_Genome: Genes or genomes
- Clinical_Drug: Clinical drugs
- Pharmacologic_Substance: Medications
- Medical_Device: Medical devices
- Anatomical_Abnormality: Anatomical abnormalities
- Amino_Acid_Peptide_or_Protein: Proteins, peptides
- Hormone: Hormones
- Enzyme: Enzymes
- Bacterium: Bacteria
- Virus: Viruses

Rules:
1. Extract EXACT text spans only
2. FOCUS on: anatomical structures, drugs, medical devices, genes/proteins
3. IGNORE: generic terms like "human", "patient", common foods (unless allergens)
4. Return valid JSON matching the schema

Text:
[INPUT TEXT]'''

CONCEPTUAL_ENTITY_PROMPT = '''Extract clinical conceptual entities from the text.

Key semantic types:
- Finding: Clinical findings
- Sign_or_Symptom: Signs or symptoms
- Laboratory_or_Test_Result: Lab or test results
- Clinical_Attribute: Clinical attributes
- Body_Location_or_Region: Body locations or regions
- Quantitative_Concept: Numerical measurements (scores, values)

Rules:
1. Extract EXACT text spans only
2. FOCUS on: findings, signs/symptoms, test results, body locations
3. IGNORE: generic terms like "study", "analysis", "data", "time", "group"
4. IGNORE temporal/spatial concepts unless critical for disease progression
5. Return valid JSON matching the schema

Text:
[INPUT TEXT]'''


CONTEXT_ENTITY_FILTER_PROMPT = """Validate clinical entities extracted from text.

INPUT:
1. Clinical text passage
2. List of entities with semantic types

VALIDATION RULES:
1. Keep entities that are:
   - Core concepts in the clinical narrative
   - Free from negation (e.g., "no evidence of", "rule out")
   - Clinically meaningful in this context

2. Remove entities that are:
   - Mentioned only in passing
   - Generic or ambiguous references
   - Duplicates with different semantic types (keep most clinically relevant)

CLINICAL TEXT:
[CLINICAL_INPUT_TEXT]

ENTITIES TO EVALUATE:
[ENTITIES_INPUT]

OUTPUT: Return ONLY valid JSON with this structure:
{
  "entities": [
    {"name": "canonical name", "semantic_type": "type", "mention": "exact text from document"}
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
