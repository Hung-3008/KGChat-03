import json

def get_15_umls_prompts():
    """
    Returns a dictionary of 15 refined "Super Prompts" based on the 
    15 official UMLS Semantic Groups.
    
    These prompts are designed to be clearer, avoid hallucinating examples,
    and reduce the "concept collision" seen in the 5-prompt version.
    """

    # ==========================================================================
    # UMLS Semantic Extraction Prompts (15-Group Refactor)
    #
    # Instructions have been refined to be stricter:
    # 1. Examples (e.g., ...) have been REMOVED from definitions to prevent
    #    the model from hallucinating them.
    # 2. Explicit constraints are added to enforce returning empty arrays
    #    and only extracting from the text.
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Prompt 1: Activities & Behaviors
    # --------------------------------------------------------------------------
    prompt_group_1_activities = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T058").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Activities & Behaviors]
[Definitions]
* T052 (Activity): An activity; a unit of work or action.
* T053 (Behavior): Overt, observable, or recordable actions or functions.
* T054 (Social Behavior): Overt activity directed toward or resulting from interaction with others.
* T055 (Individual Behavior): Overt activity of an individual.
* T056 (Daily or Recreational Activity): Activity of daily living or recreation.
* T057 (Occupational Activity): An activity associated with a job or profession.
* T058 (Health Care Activity): Activities related to the delivery of health care.
* T062 (Research Activity): Activities related to scientific investigation.
* T064 (Governmental or Regulatory Activity): Activities of a government or regulatory agency.
* T065 (Educational Activity): Activities related to teaching or learning.
* T066 (Machine Activity): An activity performed by a machine.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 2: Anatomy
    # --------------------------------------------------------------------------
    prompt_group_2_anatomy = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T017").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Anatomy]
[Definitions]
* T017 (Anatomical Structure): A component of the body.
* T018 (Embryonic Structure): An anatomical structure of an embryo.
* T021 (Fully Formed Anatomical Structure): An anatomical structure that has completed its development.
* T022 (Body System): A complex of organs that function together.
* T023 (Body Part, Organ, or Organ Component): A part of the body.
* T024 (Tissue): A collection of cells of a particular type.
* T025 (Cell): A living cell; the basic structural unit of all organisms.
* T026 (Cell Component): A component of a cell.
* T029 (Body Location or Region): A specific location or region of the body.
* T030 (Body Space or Junction): A space or junction in the body.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 3: Chemicals & Drugs
    # --------------------------------------------------------------------------
    prompt_group_3_chemicals = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T109").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Chemicals & Drugs]
[Definitions]
* T103 (Chemical): A substance composed of atoms or molecules.
* T104 (Chemical Viewed Structurally): A chemical defined by its structure.
* T109 (Organic Chemical): A chemical substance containing carbon.
* T114 (Nucleic Acid, Nucleoside, or Nucleotide): A molecule involved in genetic information.
* T116 (Amino Acid, Peptide, or Protein): A protein or its components.
* T120 (Chemical Viewed Functionally): A chemical defined by its function.
* T121 (Pharmacologic Substance): A substance with a pharmacological effect.
* T122 (Biomedical or Dental Material): A material used in medicine or dentistry.
* T123 (Biologically Active Substance): A substance that has a biological effect.
* T125 (Hormone): A hormone; a chemical messenger.
* T126 (Enzyme): An enzyme; a biological catalyst.
* T127 (Vitamin): A vitamin.
* T129 (Immunologic Factor): A substance involved in the immune response.
* T130 (Indicator, Reagent, or Diagnostic Aid): A substance used for testing or diagnosis.
* T131 (Hazardous or Poisonous Substance): A toxic or hazardous substance.
* T167 (Substance): A general material or substance.
* T192 (Receptor): A molecule (often a protein) that binds a specific substance.
* T195 (Antibiotic): A substance that kills or inhibits bacteria.
* T196 (Element, Ion, or Isotope): An element, ion, or isotope.
* T197 (Inorganic Chemical): A chemical substance not containing carbon.
* T200 (Clinical Drug): A specific therapeutic agent or branded drug.
---
[INPUT TEXT]
{insert yourtext here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 4: Concepts & Ideas
    # --------------------------------------------------------------------------
    prompt_group_4_concepts = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T077").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Concepts & Ideas]
[Definitions]
* T031 (Body Substance): A substance produced by or found in the body.
* T071 (Entity): An abstract entity (general).
* T077 (Conceptual Entity): An abstract idea or concept.
* T078 (Idea or Concept): A general idea or concept.
* T079 (Temporal Concept): A concept related to time.
* T080 (Qualitative Concept): A concept expressing a quality.
* T081 (Quantitative Concept): A concept expressing a quantity.
* T082 (Spatial Concept): A concept related to space or location.
* T089 (Regulation or Law): A rule, regulation, or law.
* T102 (Group Attribute): An attribute of a group.
* T168 (Food): Food or beverage.
* T169 (Functional Concept): A concept related to function or purpose.
* T170 (Intellectual Product): A product of human intellect.
* T171 (Language): A language.
* T185 (Classification): A classification or category.
* T201 (Clinical Attribute): An attribute related to clinical practice or observation.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 5: Devices
    # --------------------------------------------------------------------------
    prompt_group_5_devices = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T074").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Devices]
[Definitions]
* T074 (Medical Device): A device used in healthcare.
* T075 (Research Device): A device used in research.
* T203 (Drug Delivery Device): A device used to administer a drug.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 6: Disorders
    # --------------------------------------------------------------------------
    prompt_group_6_disorders = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T019").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Disorders]
[Definitions]
* T019 (Congenital Abnormality): An anatomical abnormality present at birth.
* T020 (Acquired Abnormality): An anatomical abnormality acquired after birth.
* T033 (Finding): An observation or finding from a test or examination.
* T034 (Laboratory or Test Result): The result of a laboratory or diagnostic test.
* T037 (Injury or Poisoning): Damage to the body or poisoning.
* T047 (Disease or Syndrome): A specific disease or syndrome.
* T048 (Mental or Behavioral Dysfunction): A disorder of mental or behavioral function.
* T050 (Experimental Model of Disease): An animal or other model used to study a disease.
* T184 (Sign or Symptom): An indication of a disease (subjective or objective).
* T190 (Anatomical Abnormality): A structural deviation from the normal.
* T191 (Neoplastic Process): A tumor or cancer process.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 7: Genes & Molecular Sequences
    # --------------------------------------------------------------------------
    prompt_group_7_genes = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T028").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Genes & Molecular Sequences]
[Definitions]
* T028 (Gene or Genome): A gene or the entire genome.
* T085 (Molecular Sequence): A sequence of molecules.
* T086 (Nucleotide Sequence): A sequence of nucleotides.
* T087 (Amino Acid Sequence): A sequence of amino acids.
* T088 (Carbohydrate Sequence): A sequence of carbohydrates.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 8: Geographic Areas
    # --------------------------------------------------------------------------
    prompt_group_8_geography = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T083").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Geographic Areas]
[Definitions]
* T083 (Geographic Area): A specific geographic location or area.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 9: Living Beings
    # --------------------------------------------------------------------------
    prompt_group_9_living_beings = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T001").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Living Beings]
[Definitions]
* T001 (Organism): A living being.
* T002 (Plant): A plant.
* T004 (Fungus): A fungus.
* T005 (Virus): A virus.
* T007 (Bacterium): A bacterium.
* T008 (Animal): An animal.
* T010 (Vertebrate): An animal with a backbone.
* T011 (Amphibian): An amphibian.
* T012 (Bird): A bird.
* T013 (Fish): A fish.
* T014 (Reptile): A reptile.
* T015 (Mammal): A mammal.
* T016 (Human): A human being.
* T194 (Archaeon): An archaeon.
* T204 (Eukaryote): An organism with a cell nucleus.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 10: Objects
    # --------------------------------------------------------------------------
    prompt_group_10_objects = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T072").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Objects]
[Definitions]
* T072 (Physical Object): A tangible, non-living object.
* T073 (Manufactured Object): An object created by humans.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 11: Occupations
    # --------------------------------------------------------------------------
    prompt_group_11_occupations = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T090").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Occupations]
[Definitions]
* T090 (Occupation or Discipline): A job, profession, or field of study.
* T091 (Biomedical Occupation or Discipline): A profession or field in biomedicine.
* T097 (Professional or Occupational Group): A group of people in the same profession.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 12: Organizations
    # --------------------------------------------------------------------------
    prompt_group_12_organizations = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T092").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Organizations]
[Definitions]
* T032 (Organism Attribute): An attribute of an organism.
* T092 (Organization): An organized group of people.
* T093 (Health Care Related Organization): An organization related to health care.
* T094 (Professional Society): An organization for professionals.
* T095 (Self-help or Relief Organization): An organization for self-help or relief.
* T096 (Group): A general group of individuals.
* T098 (Population Group): A group of people defined by population characteristics.
* T099 (Family Group): A family.
* T100 (Age Group): A group of people defined by age.
* T101 (Patient or Disabled Group): A group of patients or disabled persons.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 13: Phenomena
    # --------------------------------------------------------------------------
    prompt_group_13_phenomena = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T067").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Phenomena]
[Definitions]
* T051 (Event): A significant occurrence or event.
* T067 (Phenomenon or Process): A phenomenon or process.
* T068 (Human-caused Phenomenon or Process): A phenomenon caused by humans.
* T069 (Environmental Effect of Humans): The effect of human activity on the environment.
* T070 (Natural Phenomenon or Process): A phenomenon that occurs naturally.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 14: Physiology
    # --------------------------------------------------------------------------
    prompt_group_14_physiology = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T038").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Physiology]
[Definitions]
* T038 (Biologic Function): A function of a living organism.
* T039 (Physiologic Function): A normal function of the body.
* T040 (Organism Function): A function of an entire organism.
* T041 (Mental Process): A process of the mind.
* T042 (Organ or Tissue Function): The function of an organ or tissue.
* T043 (Cell Function): The function of a cell.
* T044 (Molecular Function): The function of a molecule.
* T045 (Genetic Function): The function related to genetics.
* T046 (Pathologic Function): A function associated with a disease or abnormal state.
* T049 (Cell or Molecular Dysfunction): Abnormal function at the cell or molecular level.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # --------------------------------------------------------------------------
    # Prompt 15: Procedures
    # --------------------------------------------------------------------------
    prompt_group_15_procedures = """
Task: Read the [INPUT TEXT] and extract all entities that strictly match any of the definitions provided in the [Definitions] section.

Constraints:
1.  Return a single JSON object.
2.  Each key in the JSON must be a TUI code (e.g., "T059").
3.  Each value must be an array of the exact entity strings found in the text.
4.  If no entities are found for a specific TUI, return an empty array [] for that key.
5.  Do NOT include any entities that are not explicitly present in the [INPUT TEXT].

---
[ENTITY GROUP: Procedures]
[Definitions]
* T059 (Laboratory Procedure): A procedure carried out in a laboratory.
* T060 (Diagnostic Procedure): A procedure used to diagnose a condition.
* T061 (Therapeutic or Preventive Procedure): A procedure used to treat or prevent a condition.
* T063 (Molecular Biology Research Technique): A technique used in molecular biology research.
---
[INPUT TEXT]
{insert your text here}

[OUTPUT JSON]
"""

    # Return all prompts in a dictionary for easy access
    return {
        "group_1_activities": prompt_group_1_activities,
        "group_2_anatomy": prompt_group_2_anatomy,
        "group_3_chemicals": prompt_group_3_chemicals,
        "group_4_concepts": prompt_group_4_concepts,
        "group_5_devices": prompt_group_5_devices,
        "group_6_disorders": prompt_group_6_disorders,
        "group_7_genes": prompt_group_7_genes,
        "group_8_geography": prompt_group_8_geography,
        "group_9_living_beings": prompt_group_9_living_beings,
        "group_10_objects": prompt_group_10_objects,
        "group_11_occupations": prompt_group_11_occupations,
        "group_12_organizations": prompt_group_12_organizations,
        "group_13_phenomena": prompt_group_13_phenomena,
        "group_14_physiology": prompt_group_14_physiology,
        "group_15_procedures": prompt_group_15_procedures
    }



def get_filter_prompt():
    """
    Returns the validation and consolidation prompt.
    This prompt is designed to be Step 2, taking the raw output from 
    the 15 extraction prompts and cleaning it.
    """
    
    # This prompt assumes you will format the raw_output as a JSON string
    # when you insert it into the {raw_extracted_json} placeholder.
    
    prompt_filter_and_consolidate = """
You are an expert in Biomedical Informatics and UMLS Semantic Types.
Your task is to review a list of **raw candidate entities** extracted from a text. This raw list is known to contain errors, hallucinations, duplicates, and severe misclassifications.

You must follow these 3 critical rules to produce a final, clean list.

---
[RULES]

1.  **VALIDATE (No Hallucinations):** For every candidate, you MUST check if the `name` actually exists in the [ORIGINAL TEXT]. If the `name` is not in the text (e.g., "Vitamin", "Protein", "surgery"), you MUST discard it.

2.  **VERIFY (No Misclassification):** If the `name` is in the text, you MUST evaluate its `semantic_type`. Is this type a logical and specific fit?
    * **Bad:** 'mosquito' as 'Food' -> DISCARD.
    * **Bad:** 'diabetes' as 'Gene' -> DISCARD.
    * **Good:** 'diabetes' as 'Disease or Syndrome' -> KEEP.
    * **Good:** 'thirst' as 'Sign or Symptom' -> KEEP.

3.  **CONSOLIDATE (No Duplicates):** An entity should only appear ONCE in the final list. If the same entity (e.g., "diabetes") appears multiple times in the raw list, you must select the **single most accurate** classification for it (e.g., 'Disease or Syndrome') and discard all other (incorrect) versions.

---
[ORIGINAL TEXT]
{insert_original_text_here}

---
[RAW CANDIDATE LIST]
{insert_raw_extracted_json_here}

---
[FINAL CLEANED JSON]
Task: Based on the 3 rules above, produce the final, clean JSON output.
The output must be a single JSON array of objects, each with a "name" and "semantic_type".

Example output format:
[
  {"name": "diabetes", "semantic_type": "Disease or Syndrome"},
  {"name": "dengue fever", "semantic_type": "Disease or Syndrome"},
  {"name": "thirst and polyuria", "semantic_type": "Sign or Symptom"},
  ...
]

[OUTPUT]
"""
    return prompt_filter_and_consolidate

if __name__ == "__main__":
    # Example of how to get and use the prompts
    all_prompts = get_15_umls_prompts()
    
    print(f"Successfully generated {len(all_prompts)} prompts.")
    print("\n--- EXAMPLE: PROMPT 6 (Disorders) ---")
    
    # Example text to be inserted
    my_text = "The patient presents with fever and a persistent cough. An X-ray ruled out pneumonia."
    
    # Get the prompt template
    prompt_template = all_prompts["group_6_disorders"]
    
    # Insert the text
    final_prompt = prompt_template.format(insert_your_text_here=my_text)
    
    print(final_prompt)
    
    # In a real application, you would send `final_prompt` to the LLM.
    # The expected LLM output for this example would be:
    # {
    #   "T019": [],
    #   "T020": [],
    #   "T033": ["fever", "persistent cough"],
    #   "T034": [],
    #   "T037": [],
    #   "T047": ["pneumonia"],
    #   "T048": [],
    #   "T050": [],
    #   "T184": ["fever", "persistent cough"],
    #   "T190": [],
    #   "T191": []
    # }