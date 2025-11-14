from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ExtractedEntity(BaseModel):
    """A single extracted entity.

    - name: canonical name or identifier
    - mention: optional surface text from the input that triggered the extraction
    """
    name: str = Field(..., description="Canonical name or identifier of the entity")
    mention: Optional[str] = Field(default=None, description="Surface mention in the input text")
    semantic_type: str = Field(..., description="UMLS TUI code for the semantic type")


class ActivityOutput(BaseModel):
    """Schema for ACTIVITY_PROMPT extraction results.
    
    Contains lists of extracted entities grouped by semantic category (activity type).
    Each category field contains a list of entity names that belong to that semantic type.
    """
    behavior: List[str] = Field(default_factory=list, description="Category T053 (Behavior): Entities describing general behavioral activities")
    social_behavior: List[str] = Field(default_factory=list, description="Category T054 (Social Behavior): Entities describing social or interpersonal behaviors")
    individual_behavior: List[str] = Field(default_factory=list, description="Category T055 (Individual Behavior): Entities describing individual personal behaviors")
    daily_or_recreational_activity: List[str] = Field(default_factory=list, description="Category T056 (Daily or Recreational Activity): Entities describing everyday or leisure activities")
    occupational_activity: List[str] = Field(default_factory=list, description="Category T057 (Occupational Activity): Entities describing work-related activities")
    health_care_activity: List[str] = Field(default_factory=list, description="Category T058 (Health Care Activity): Entities describing healthcare-related activities")
    laboratory_procedure: List[str] = Field(default_factory=list, description="Category T059 (Laboratory Procedure): Entities describing laboratory procedures and tests")
    diagnostic_procedure: List[str] = Field(default_factory=list, description="Category T060 (Diagnostic Procedure): Entities describing diagnostic procedures and examinations")
    therapeutic_or_preventive_procedure: List[str] = Field(default_factory=list, description="Category T061 (Therapeutic or Preventive Procedure): Entities describing treatment or preventive procedures")
    research_activity: List[str] = Field(default_factory=list, description="Category T062 (Research Activity): Entities describing research activities and studies")
    molecular_biology_research_technique: List[str] = Field(default_factory=list, description="Category T063 (Molecular Biology Research Technique): Entities describing molecular biology research methods")
    governmental_or_regulatory_activity: List[str] = Field(default_factory=list, description="Category T064 (Governmental or Regulatory Activity): Entities describing regulatory or governmental activities")
    educational_activity: List[str] = Field(default_factory=list, description="Category T065 (Educational Activity): Entities describing educational or training activities")
    machine_activity: List[str] = Field(default_factory=list, description="Category T066 (Machine Activity): Entities describing machine or automated activities")


class PhenomenonOutput(BaseModel):
    """Schema for PHENOMENON_PROMPT extraction results.
    
    Contains lists of extracted entities grouped by semantic category (phenomenon/process type).
    Each category field contains a list of entity names that belong to that semantic type.
    """
    injury_or_poisoning: List[str] = Field(default_factory=list, description="Category T037 (Injury or Poisoning): Entities describing injuries or poisoning incidents")
    human_caused_phenomenon_or_process: List[str] = Field(default_factory=list, description="Category T068 (Human-caused Phenomenon or Process): Entities describing phenomena or processes caused by humans")
    environmental_effect_of_humans: List[str] = Field(default_factory=list, description="Category T069 (Environmental Effect of Humans): Entities describing environmental effects caused by human activities")
    natural_phenomenon_or_process: List[str] = Field(default_factory=list, description="Category T070 (Natural Phenomenon or Process): Entities describing natural phenomena or processes")
    biologic_function: List[str] = Field(default_factory=list, description="Category T038 (Biologic Function): Entities describing biological functions")
    physiologic_function: List[str] = Field(default_factory=list, description="Category T039 (Physiologic Function): Entities describing physiological functions and processes")
    organism_function: List[str] = Field(default_factory=list, description="Category T040 (Organism Function): Entities describing overall organism functions")
    mental_process: List[str] = Field(default_factory=list, description="Category T041 (Mental Process): Entities describing mental or cognitive processes")
    organ_or_tissue_function: List[str] = Field(default_factory=list, description="Category T042 (Organ or Tissue Function): Entities describing functions of organs or tissues")
    cell_function: List[str] = Field(default_factory=list, description="Category T043 (Cell Function): Entities describing cellular functions")
    molecular_function: List[str] = Field(default_factory=list, description="Category T044 (Molecular Function): Entities describing molecular-level functions")
    genetic_function: List[str] = Field(default_factory=list, description="Category T045 (Genetic Function): Entities describing genetic functions and processes")
    pathologic_function: List[str] = Field(default_factory=list, description="Category T046 (Pathologic Function): Entities describing abnormal or pathological functions")
    disease_or_syndrome: List[str] = Field(default_factory=list, description="Category T047 (Disease or Syndrome): Entities describing diseases or syndromes")
    mental_or_behavioral_dysfunction: List[str] = Field(default_factory=list, description="Category T048 (Mental or Behavioral Dysfunction): Entities describing mental or behavioral disorders")
    neoplastic_process: List[str] = Field(default_factory=list, description="Category T191 (Neoplastic Process): Entities describing neoplastic or cancerous processes")
    cell_or_molecular_dysfunction: List[str] = Field(default_factory=list, description="Category T049 (Cell or Molecular Dysfunction): Entities describing cellular or molecular dysfunctions")
    experimental_model_of_disease: List[str] = Field(default_factory=list, description="Category T050 (Experimental Model of Disease): Entities describing experimental disease models")


class PhysicalObjectOutput(BaseModel):
    """Schema for PHYSICAL_OBJECT_PROMPT extraction results.
    
    Contains lists of extracted entities grouped by semantic category (physical object type).
    Each category field contains a list of entity names that belong to that semantic type.
    """
    # Organisms
    organism: List[str] = Field(default_factory=list, description="Category T001 (Organism): Entities describing organisms")
    virus: List[str] = Field(default_factory=list, description="Category T005 (Virus): Entities describing viruses")
    bacterium: List[str] = Field(default_factory=list, description="Category T007 (Bacterium): Entities describing bacteria")
    archaeon: List[str] = Field(default_factory=list, description="Category T194 (Archaeon): Entities describing archaea")
    eukaryote: List[str] = Field(default_factory=list, description="Category T204 (Eukaryote): Entities describing eukaryotes")
    plant: List[str] = Field(default_factory=list, description="Category T002 (Plant): Entities describing plants")
    fungus: List[str] = Field(default_factory=list, description="Category T004 (Fungus): Entities describing fungi")
    animal: List[str] = Field(default_factory=list, description="Category T008 (Animal): Entities describing animals")
    vertebrate: List[str] = Field(default_factory=list, description="Category T010 (Vertebrate): Entities describing vertebrates")
    amphibian: List[str] = Field(default_factory=list, description="Category T011 (Amphibian): Entities describing amphibians")
    bird: List[str] = Field(default_factory=list, description="Category T012 (Bird): Entities describing birds")
    fish: List[str] = Field(default_factory=list, description="Category T013 (Fish): Entities describing fish")
    reptile: List[str] = Field(default_factory=list, description="Category T014 (Reptile): Entities describing reptiles")
    mammal: List[str] = Field(default_factory=list, description="Category T015 (Mammal): Entities describing mammals")
    human: List[str] = Field(default_factory=list, description="Category T016 (Human): Entities describing humans")
    
    # Anatomical Structures
    anatomical_structure: List[str] = Field(default_factory=list, description="Category T017 (Anatomical Structure): Entities describing anatomical structures")
    embryonic_structure: List[str] = Field(default_factory=list, description="Category T018 (Embryonic Structure): Entities describing embryonic structures")
    fully_formed_anatomical_structure: List[str] = Field(default_factory=list, description="Category T021 (Fully Formed Anatomical Structure): Entities describing fully developed anatomical structures")
    body_part_organ_or_organ_component: List[str] = Field(default_factory=list, description="Category T023 (Body Part, Organ, or Organ Component): Entities describing body parts, organs, or organ components")
    tissue: List[str] = Field(default_factory=list, description="Category T024 (Tissue): Entities describing tissues")
    cell: List[str] = Field(default_factory=list, description="Category T025 (Cell): Entities describing cells")
    cell_component: List[str] = Field(default_factory=list, description="Category T026 (Cell Component): Entities describing cellular components")
    gene_or_genome: List[str] = Field(default_factory=list, description="Category T028 (Gene or Genome): Entities describing genes or genomes")
    anatomical_abnormality: List[str] = Field(default_factory=list, description="Category T190 (Anatomical Abnormality): Entities describing anatomical abnormalities")
    congenital_abnormality: List[str] = Field(default_factory=list, description="Category T019 (Congenital Abnormality): Entities describing congenital abnormalities")
    acquired_abnormality: List[str] = Field(default_factory=list, description="Category T020 (Acquired Abnormality): Entities describing acquired abnormalities")
    
    # Manufactured Objects
    manufactured_object: List[str] = Field(default_factory=list, description="Category T073 (Manufactured Object): Entities describing manufactured objects")
    medical_device: List[str] = Field(default_factory=list, description="Category T074 (Medical Device): Entities describing medical devices")
    drug_delivery_device: List[str] = Field(default_factory=list, description="Category T203 (Drug Delivery Device): Entities describing drug delivery devices")
    research_device: List[str] = Field(default_factory=list, description="Category T075 (Research Device): Entities describing research devices")
    clinical_drug: List[str] = Field(default_factory=list, description="Category T200 (Clinical Drug): Entities describing clinical drugs")
    
    # Substances
    substance: List[str] = Field(default_factory=list, description="Category T167 (Substance): Entities describing substances")
    body_substance: List[str] = Field(default_factory=list, description="Category T031 (Body Substance): Entities describing body substances")
    chemical: List[str] = Field(default_factory=list, description="Category T103 (Chemical): Entities describing chemicals")
    chemical_viewed_structurally: List[str] = Field(default_factory=list, description="Category T104 (Chemical Viewed Structurally): Entities describing chemicals by their structural properties")
    organic_chemical: List[str] = Field(default_factory=list, description="Category T109 (Organic Chemical): Entities describing organic chemicals")
    nucleic_acid_nucleoside_or_nucleotide: List[str] = Field(default_factory=list, description="Category T114 (Nucleic Acid, Nucleoside, or Nucleotide): Entities describing nucleic acids and related compounds")
    amino_acid_peptide_or_protein: List[str] = Field(default_factory=list, description="Category T116 (Amino Acid, Peptide, or Protein): Entities describing proteins and related compounds")
    element_ion_or_isotope: List[str] = Field(default_factory=list, description="Category T196 (Element, Ion, or Isotope): Entities describing elements and their forms")
    inorganic_chemical: List[str] = Field(default_factory=list, description="Category T197 (Inorganic Chemical): Entities describing inorganic chemicals")
    chemical_viewed_functionally: List[str] = Field(default_factory=list, description="Category T120 (Chemical Viewed Functionally): Entities describing chemicals by their functional properties")
    pharmacologic_substance: List[str] = Field(default_factory=list, description="Category T121 (Pharmacologic Substance): Entities describing pharmacologic substances")
    antibiotic: List[str] = Field(default_factory=list, description="Category T195 (Antibiotic): Entities describing antibiotics")
    biomedical_or_dental_material: List[str] = Field(default_factory=list, description="Category T122 (Biomedical or Dental Material): Entities describing biomedical or dental materials")
    biologically_active_substance: List[str] = Field(default_factory=list, description="Category T123 (Biologically Active Substance): Entities describing biologically active substances")
    hormone: List[str] = Field(default_factory=list, description="Category T125 (Hormone): Entities describing hormones")
    enzyme: List[str] = Field(default_factory=list, description="Category T126 (Enzyme): Entities describing enzymes")
    vitamin: List[str] = Field(default_factory=list, description="Category T127 (Vitamin): Entities describing vitamins")
    immunologic_factor: List[str] = Field(default_factory=list, description="Category T129 (Immunologic Factor): Entities describing immunologic factors")
    receptor: List[str] = Field(default_factory=list, description="Category T192 (Receptor): Entities describing receptors")
    indicator_reagent_or_diagnostic_aid: List[str] = Field(default_factory=list, description="Category T130 (Indicator, Reagent, or Diagnostic Aid): Entities describing diagnostic indicators and reagents")
    hazardous_or_poisonous_substance: List[str] = Field(default_factory=list, description="Category T131 (Hazardous or Poisonous Substance): Entities describing hazardous or poisonous substances")
    food: List[str] = Field(default_factory=list, description="Category T168 (Food): Entities describing food items")


class ConceptualEntityOutput(BaseModel):
    """Schema for CONCEPTUAL_ENTITY_PROMPT extraction results.
    
    Contains lists of extracted entities grouped by semantic category (conceptual entity type).
    Each category field contains a list of entity names that belong to that semantic type.
    """
    # Organism Attribute & Finding
    organism_attribute: List[str] = Field(default_factory=list, description="Category T032 (Organism Attribute): Entities describing attributes of organisms")
    clinical_attribute: List[str] = Field(default_factory=list, description="Category T201 (Clinical Attribute): Entities describing clinical attributes")
    finding: List[str] = Field(default_factory=list, description="Category T033 (Finding): Entities describing clinical findings")
    laboratory_or_test_result: List[str] = Field(default_factory=list, description="Category T034 (Laboratory or Test Result): Entities describing laboratory or test results")
    sign_or_symptom: List[str] = Field(default_factory=list, description="Category T184 (Sign or Symptom): Entities describing clinical signs or symptoms")
    
    # Idea or Concept
    idea_or_concept: List[str] = Field(default_factory=list, description="Category T078 (Idea or Concept): Entities describing ideas or concepts")
    temporal_concept: List[str] = Field(default_factory=list, description="Category T079 (Temporal Concept): Entities describing time-related concepts")
    qualitative_concept: List[str] = Field(default_factory=list, description="Category T080 (Qualitative Concept): Entities describing qualitative concepts")
    quantitative_concept: List[str] = Field(default_factory=list, description="Category T081 (Quantitative Concept): Entities describing quantitative or measurement concepts")
    spatial_concept: List[str] = Field(default_factory=list, description="Category T082 (Spatial Concept): Entities describing spatial concepts")
    body_location_or_region: List[str] = Field(default_factory=list, description="Category T029 (Body Location or Region): Entities describing body locations or regions")
    body_space_or_junction: List[str] = Field(default_factory=list, description="Category T030 (Body Space or Junction): Entities describing body spaces or junctions")
    geographic_area: List[str] = Field(default_factory=list, description="Category T083 (Geographic Area): Entities describing geographic areas")
    molecular_sequence: List[str] = Field(default_factory=list, description="Category T085 (Molecular Sequence): Entities describing molecular sequences")
    nucleotide_sequence: List[str] = Field(default_factory=list, description="Category T086 (Nucleotide Sequence): Entities describing nucleotide sequences")
    amino_acid_sequence: List[str] = Field(default_factory=list, description="Category T087 (Amino Acid Sequence): Entities describing amino acid sequences")
    carbohydrate_sequence: List[str] = Field(default_factory=list, description="Category T088 (Carbohydrate Sequence): Entities describing carbohydrate sequences")
    functional_concept: List[str] = Field(default_factory=list, description="Category T169 (Functional Concept): Entities describing functional concepts")
    body_system: List[str] = Field(default_factory=list, description="Category T022 (Body System): Entities describing body systems")
    
    # Occupation, Discipline, Organization & Groups
    occupation_or_discipline: List[str] = Field(default_factory=list, description="Category T090 (Occupation or Discipline): Entities describing occupations or disciplines")
    biomedical_occupation_or_discipline: List[str] = Field(default_factory=list, description="Category T091 (Biomedical Occupation or Discipline): Entities describing biomedical occupations or disciplines")
    organization: List[str] = Field(default_factory=list, description="Category T092 (Organization): Entities describing organizations")
    health_care_related_organization: List[str] = Field(default_factory=list, description="Category T093 (Health Care Related Organization): Entities describing healthcare-related organizations")
    professional_society: List[str] = Field(default_factory=list, description="Category T094 (Professional Society): Entities describing professional societies")
    self_help_or_relief_organization: List[str] = Field(default_factory=list, description="Category T095 (Self-help or Relief Organization): Entities describing self-help or relief organizations")
    group: List[str] = Field(default_factory=list, description="Category T096 (Group): Entities describing groups")
    professional_or_occupational_group: List[str] = Field(default_factory=list, description="Category T097 (Professional or Occupational Group): Entities describing professional or occupational groups")
    population_group: List[str] = Field(default_factory=list, description="Category T098 (Population Group): Entities describing population groups")
    family_group: List[str] = Field(default_factory=list, description="Category T099 (Family Group): Entities describing family groups")
    age_group: List[str] = Field(default_factory=list, description="Category T100 (Age Group): Entities describing age groups")
    patient_or_disabled_group: List[str] = Field(default_factory=list, description="Category T101 (Patient or Disabled Group): Entities describing patient or disabled groups")
    group_attribute: List[str] = Field(default_factory=list, description="Category T102 (Group Attribute): Entities describing group attributes")
    
    # Intellectual Product
    intellectual_product: List[str] = Field(default_factory=list, description="Category T170 (Intellectual Product): Entities describing intellectual products")
    regulation_or_law: List[str] = Field(default_factory=list, description="Category T089 (Regulation or Law): Entities describing regulations or laws")
    classification: List[str] = Field(default_factory=list, description="Category T185 (Classification): Entities describing classification systems")
    language: List[str] = Field(default_factory=list, description="Category T171 (Language): Entities describing languages")


class ValidatedEntity(BaseModel):
    """A validated entity after filtering."""
    name: str = Field(..., description="Entity name")
    semantic_type: str = Field(..., description="Semantic type field name")
    cluster: str = Field(..., description="Cluster name (activity, phenomenon, physical_object, conceptual_entity)")


class FilterOutput(BaseModel):
    """Schema for filter/validation output."""
    entities: List[ValidatedEntity] = Field(default_factory=list, description="List of validated entities")


__all__ = [
    'ExtractedEntity',
    'ActivityOutput',
    'PhenomenonOutput',
    'PhysicalObjectOutput',
    'ConceptualEntityOutput',
    'ValidatedEntity',
    'FilterOutput'
]
