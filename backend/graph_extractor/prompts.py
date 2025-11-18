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


ACTIVITY_PROMPT = '''You are an expert clinical entity extraction system. Your task is to extract entities from clinical text and categorize them into the appropriate activity types.

The "activity" cluster includes these semantic types:
- Activity: General activities
- Behavior: General behaviors
- Social_Behavior: Social interactions and behaviors
- Individual_Behavior: Personal behaviors
- Daily_or_Recreational_Activity: Daily routines, hobbies, leisure activities
- Occupational_Activity: Work-related activities
- Health_Care_Activity: Healthcare-related activities
- Laboratory_Procedure: Lab tests and procedures
- Diagnostic_Procedure: Diagnostic tests and examinations
- Therapeutic_or_Preventive_Procedure: Treatments, therapies, preventive measures
- Research_Activity: Research-related activities
- Molecular_Biology_Research_Technique: Molecular biology research methods
- Governmental_or_Regulatory_Activity: Regulatory, governmental activities
- Educational_Activity: Educational activities, teaching, learning
- Machine_Activity: Machine operations, automated processes

Instructions:
1. Extract EXACT text spans as they appear in the document
2. Categorize each entity into the MOST SPECIFIC semantic type that matches
3. If NO entities are found for a semantic type, return an empty list for that field
4. Return ONLY valid JSON in the exact format specified by the schema
5. DO NOT include any additional text, explanations, or formatting outside the JSON structure
6. EXTRACT ONLY ATOMIC ENTITIES - Never extract phrases containing multiple activities or actions combined with medications, devices, or other non-activity elements.
   - CORRECT: "physical therapy", "chest X-ray", "blood test"
   - INCORRECT: "takes metformin daily", "underwent surgery with anesthesia", "prescribed antibiotics"
Text:
[INPUT TEXT]
'''

PHENOMENON_PROMPT = '''You are an expert clinical entity extraction system. Your task is to extract entities from clinical text and categorize them into the appropriate phenomenon types.

The "phenomenon" cluster includes these semantic types:
- Phenomenon_or_Process: General phenomena or processes
- Injury_or_Poisoning: Injuries or poisonings
- Human_caused_Phenomenon_or_Process: Phenomena or processes caused by humans
- Environmental_Effect_of_Humans: Environmental effects caused by human activities
- Natural_Phenomenon_or_Process: Natural phenomena or processes
- Biologic_Function: Biological functions
- Physiologic_Function: Physiological functions
- Organism_Function: Functions of organisms
- Mental_Process: Mental processes
- Organ_or_Tissue_Function: Functions of organs or tissues
- Cell_Function: Cellular functions
- Molecular_Function: Molecular functions
- Genetic_Function: Genetic functions
- Cell_or_Molecular_Dysfunction: Dysfunctions at cellular or molecular level
- Pathologic_Function: Pathological functions
- Disease_or_Syndrome: Diseases or syndromes
- Mental_or_Behavioral_Dysfunction: Mental or behavioral dysfunctions
- Neoplastic_Process: Neoplastic processes (tumor/cancer-related)
- Experimental_Model_of_Disease: Experimental models of diseases

Instructions:
1. Extract EXACT text spans as they appear in the document
2. Categorize each entity into the MOST SPECIFIC semantic type that matches
3. If NO entities are found for a semantic type, return an empty list for that field
4. Return ONLY valid JSON in the exact format specified by the schema
5. DO NOT include any additional text, explanations, or formatting outside the JSON structure

Text:
[INPUT TEXT]
'''

PHYSICAL_OBJECT_PROMPT = '''You are an expert clinical entity extraction system. Your task is to extract entities from clinical text and categorize them into the appropriate physical object types.

The "physical_object" cluster includes these semantic types:
- Physical_Object: General physical objects
- Organism: Living organisms
- Virus: Viruses
- Bacterium: Bacteria
- Archaeon: Archaeons
- Eukaryote: Eukaryotes
- Plant: Plants
- Fungus: Fungi
- Animal: Animals
- Vertebrate: Vertebrates
- Amphibian: Amphibians
- Bird: Birds
- Fish: Fish
- Reptile: Reptiles
- Mammal: Mammals
- Human: Humans
- Anatomical_Structure: Anatomical structures
- Embryonic_Structure: Embryonic structures
- Fully_Formed_Anatomical_Structure: Fully formed anatomical structures
- Body_Part_Organ_or_Organ_Component: Body parts, organs, or organ components
- Tissue: Tissues
- Cell: Cells
- Cell_Component: Cell components
- Gene_or_Genome: Genes or genomes
- Anatomical_Abnormality: Anatomical abnormalities
- Congenital_Abnormality: Congenital abnormalities
- Acquired_Abnormality: Acquired abnormalities
- Manufactured_Object: Manufactured objects
- Medical_Device: Medical devices
- Drug_Delivery_Device: Drug delivery devices
- Research_Device: Research devices
- Clinical_Drug: Clinical drugs
- Substance: Substances
- Body_Substance: Body substances
- Chemical: Chemicals
- Chemical_Viewed_Structurally: Chemicals viewed structurally
- Organic_Chemical: Organic chemicals
- Nucleic_Acid_Nucleoside_or_Nucleotide: Nucleic acids, nucleosides, or nucleotides
- Amino_Acid_Peptide_or_Protein: Amino acids, peptides, or proteins
- Element_Ion_or_Isotope: Elements, ions, or isotopes
- Inorganic_Chemical: Inorganic chemicals
- Chemical_Viewed_Functionally: Chemicals viewed functionally
- Pharmacologic_Substance: Pharmacologic substances
- Antibiotic: Antibiotics
- Biomedical_or_Dental_Material: Biomedical or dental materials
- Biologically_Active_Substance: Biologically active substances
- Hormone: Hormones
- Enzyme: Enzymes
- Vitamin: Vitamins
- Immunologic_Factor: Immunologic factors
- Receptor: Receptors
- Indicator_Reagent_or_Diagnostic_Aid: Indicator reagents or diagnostic aids
- Hazardous_or_Poisonous_Substance: Hazardous or poisonous substances
- Food: Foods

Instructions:
1. Extract EXACT text spans as they appear in the document
2. Categorize each entity into the MOST SPECIFIC semantic type that matches
3. If NO entities are found for a semantic type, return an empty list for that field
4. Return ONLY valid JSON in the exact format specified by the schema
5. DO NOT include any additional text, explanations, or formatting outside the JSON structure

Text:
[INPUT TEXT]'''

CONCEPTUAL_ENTITY_PROMPT = '''You are an expert clinical entity extraction system. Your task is to extract entities from clinical text and categorize them into the appropriate conceptual entity types.

The "conceptual_entity" cluster includes these semantic types:
- Conceptual_Entity: General conceptual entities
- Organism_Attribute: Attributes of organisms
- Clinical_Attribute: Clinical attributes
- Finding: Findings
- Laboratory_or_Test_Result: Laboratory or test results
- Sign_or_Symptom: Signs or symptoms
- Idea_or_Concept: Ideas or concepts
- Temporal_Concept: Temporal concepts (time-related)
- Qualitative_Concept: Qualitative concepts (descriptive qualities)
- Quantitative_Concept: Quantitative concepts (numerical measurements)
- Spatial_Concept: Spatial concepts (location, position, direction)
- Body_Location_or_Region: Body locations or regions
- Body_Space_or_Junction: Body spaces or junctions
- Geographic_Area: Geographic areas
- Molecular_Sequence: Molecular sequences
- Nucleotide_Sequence: Nucleotide sequences
- Amino_Acid_Sequence: Amino acid sequences
- Carbohydrate_Sequence: Carbohydrate sequences
- Functional_Concept: Functional concepts
- Body_System: Body systems
- Occupation_or_Discipline: Occupations or disciplines
- Biomedical_Occupation_or_Discipline: Biomedical occupations or disciplines
- Organization: Organizations
- Health_Care_Related_Organization: Health care related organizations
- Professional_Society: Professional societies
- Self_help_or_Relief_Organization: Self-help or relief organizations
- Group: Groups
- Professional_or_Occupational_Group: Professional or occupational groups
- Population_Group: Population groups
- Family_Group: Family groups
- Age_Group: Age groups
- Patient_or_Disabled_Group: Patient or disabled groups
- Group_Attribute: Group attributes
- Intellectual_Product: Intellectual products
- Regulation_or_Law: Regulations or laws
- Classification: Classifications
- Language: Languages

Instructions:
1. Extract EXACT text spans as they appear in the document
2. Categorize each entity into the MOST SPECIFIC semantic type that matches
3. If NO entities are found for a semantic type, return an empty list for that field
4. Return ONLY valid JSON in the exact format specified by the schema
5. DO NOT include any additional text, explanations, or formatting outside the JSON structure
Text:
[INPUT TEXT]'''


CONTEXT_ENTITY_FILTER_PROMPT = """You are an expert clinical entity validation system. Your task is to review entities that have already passed hierarchical filtering and make final determinations about which entities should be kept based on clinical context and semantic appropriateness.

## INPUT STRUCTURE
You will receive:
1. The original clinical text passage
2. A list of entities grouped by semantic cluster and semantic type
3. Information about which entities share the same hierarchical depth (same specificity level)

## FILTERING CRITERIA
For each entity that shares the same hierarchical depth with other entities, evaluate whether it should be kept based on:

1. CONTEXTUAL APPROPRIATENESS:
   - Does the entity appear as a core concept in the clinical narrative (not just mentioned in passing)?
   - Is the entity explicitly relevant to the patient's case described in the text?
   - Is the entity free from negation or uncertainty markers (e.g., "no evidence of", "rule out", "possible") unless specifically about ruling out conditions?

2. SEMANTIC FIT:
   - Does the entity truly belong to its assigned semantic type in this specific context?
   - For example: "Aspirin" is a medication (Clinical_Drug) but if mentioned in the context of "aspirin allergy", it becomes an allergen (not a treatment).

3. CLINICAL SIGNIFICANCE:
   - Is the entity clinically meaningful in this context?
   - Does it represent a direct observation or intervention rather than a general reference?

## SPECIFIC RULES
1. For entities with identical text but different semantic types at the same depth:
   - Keep only the entity with the most clinically relevant semantic type for this context
   - Example: "MRI" could be a Diagnostic_Procedure or a Machine_Activity, but in clinical context, Diagnostic_Procedure is more relevant

2. For entity phrases that contain other entities:
   - Keep the more specific phrase if it provides additional clinical context
   - Example: Keep "lesion in the left temporal lobe" instead of just "lesion" if both exist at same depth
   - Exception: Keep the simpler entity if the longer phrase includes non-clinical modifiers

3. For ambiguous entities:
   - When uncertain, prefer keeping entities that directly relate to patient diagnosis, treatment, or symptoms
   - Remove entities that are merely examples, hypothetical scenarios, or general knowledge references


- Include ONLY entities that pass your validation
- Preserve the exact text of entities as they appear in the input
- If an ENTIRE semantic type has no valid entities, omit it completely (don't include empty arrays)
- In removal_explanations, document each removed entity with a concise reason (max 15 words)
- Extract the mention text for each entity from the original clinical text

## CLINICAL TEXT
[CLINICAL_INPUT_TEXT]

## ENTITIES TO EVALUATE
[ENTITIES_INPUT]

## OUTPUT INSTRUCTIONS
Return ONLY a valid JSON objec

## REMEMBER
- Focus on CLINICAL RELEVANCE in THIS SPECIFIC CONTEXT
- Be STRICT about entity boundaries and semantic type appropriateness
- When in doubt about ambiguous entities, prioritize those directly related to patient care
- DO NOT invent new entities or modify existing entity text
- DO NOT include any text outside the required JSON structure
"""

EDGE_EXTRACTION_PROMPT = """"
You are a medical knowledge extraction specialist. Analyze the clinical text below and extract **ONLY** explicit relationships between the provided entities. Follow these rules:

1. **STRICT ENTITY MATCHING**: Only use entity names EXACTLY as provided in the node list. Never modify names or introduce new entities.
2. **EXPLICIT RELATIONSHIPS ONLY**: Extract relationships DIRECTLY stated in the text. Never infer implicit connections (e.g., drug-disease treatment links without explicit text evidence).
3. **RELATION PHRASING**: Use 1-3 word relation phrases that mirror the text's verbs/prepositions (e.g., "showed", "revealed", "diagnosed with").
4. **ONE RELATION PER EDGE**: Each edge must represent a single textual relationship between one source and one target entity.
5. **NO CONFIDENCE SCORES**: Omit the 'confidence' field entirely as it's optional per schema.

**Clinical Text**:
[INPUT TEXT]

**Provided Entities** (use EXACT names):
[ENTITIES LIST]

**Output Requirements**:
- Return ONLY valid JSON matching this schema: {"edges": [{"source": "entity1", "target": "entity2", "relation": "phrase"}]}
- Include ONLY edges with explicit textual support
- Omit all other text, explanations, or fields

**Examples of VALID edges from this text**:
- MRI → lesion in the left temporal lobe (relation: "showed")
- physical examination → mild edema in the lower extremities (relation: "showed")

**Examples of INVALID edges** (DO NOT INCLUDE):
- metformin → type 2 diabetes mellitus (no explicit treatment link stated)
- C-reactive protein → hemoglobin A1c (no direct relationship described)
- Escherichia coli infection → urine culture (urine culture not in entity list)
"""