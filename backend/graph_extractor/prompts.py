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


COMBINED_ENTITY_PROMPT = '''You are a medical entity extraction expert. Extract clinical entities from the text and organize them into four clusters: activity, phenomenon, physical_object, conceptual_entity.

Rules:
- Extract exact spans from the text, prefer the most specific semantic type.
- Ignore generic/non-clinical items, keep clinically meaningful concepts.
- Return valid JSON following the provided schema.

Clusters and semantic types:
1) activity: Laboratory_Procedure, Diagnostic_Procedure, Therapeutic_or_Preventive_Procedure, Health_Care_Activity, Research_Activity, Molecular_Biology_Research_Technique, Educational_Activity, Governmental_or_Regulatory_Activity, Machine_Activity, Daily_or_Recreational_Activity, Occupational_Activity, Activity, Behavior, Social_Behavior, Individual_Behavior
2) phenomenon: Disease_or_Syndrome, Neoplastic_Process, Injury_or_Poisoning, Pathologic_Function, Mental_or_Behavioral_Dysfunction, Phenomenon_or_Process, Human_caused_Phenomenon_or_Process, Environmental_Effect_of_Humans, Natural_Phenomenon_or_Process, Biologic_Function, Physiologic_Function, Organism_Function, Organ_or_Tissue_Function, Cell_Function, Molecular_Function, Genetic_Function, Cell_or_Molecular_Dysfunction, Experimental_Model_of_Disease, Mental_Process
3) physical_object: Body_Part_Organ_or_Organ_Component, Tissue, Cell, Cell_Component, Gene_or_Genome, Anatomical_Abnormality, Congenital_Abnormality, Acquired_Abnormality, Clinical_Drug, Pharmacologic_Substance, Antibiotic, Medical_Device, Drug_Delivery_Device, Research_Device, Indicator_Reagent_or_Diagnostic_Aid, Biologically_Active_Substance, Hormone, Enzyme, Vitamin, Immunologic_Factor, Receptor, Chemical_Viewed_Structurally, Chemical_Viewed_Functionally, Organic_Chemical, Inorganic_Chemical, Amino_Acid_Peptide_or_Protein, Nucleic_Acid_Nucleoside_or_Nucleotide, Element_Ion_or_Isotope, Substance, Body_Substance, Manufactured_Object, Physical_Object, Organism, Virus, Bacterium, Archaeon, Eukaryote, Plant, Fungus, Animal, Vertebrate, Amphibian, Bird, Fish, Reptile, Mammal, Human, Food, Research_Device
4) conceptual_entity: Finding, Sign_or_Symptom, Laboratory_or_Test_Result, Clinical_Attribute, Body_Location_or_Region, Body_Space_or_Junction, Body_System, Quantitative_Concept, Qualitative_Concept, Temporal_Concept, Spatial_Concept, Functional_Concept, Molecular_Sequence, Nucleotide_Sequence, Amino_Acid_Sequence, Carbohydrate_Sequence, Organism_Attribute, Conceptual_Entity, Idea_or_Concept, Population_Group, Patient_or_Disabled_Group, Age_Group, Group, Professional_or_Occupational_Group, Biomedical_Occupation_or_Discipline, Health_Care_Related_Organization, Organization, Classification, Regulation_or_Law, Intellectual_Product, Language, Geographic_Area

Output JSON schema (strict):
{
  "activity": {"SemanticType": ["entity", ...], ...},
  "phenomenon": {"SemanticType": ["entity", ...], ...},
  "physical_object": {"SemanticType": ["entity", ...], ...},
  "conceptual_entity": {"SemanticType": ["entity", ...], ...}
}

Text:
[INPUT TEXT]
'''


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
