from pydantic import BaseModel, Field
from typing import List, Optional


# Mapping TUI codes to semantic type names for better output
TUI_TO_SEMTYPE = {
    "T204": "Eukaryote", "T002": "Plant", "T004": "Fungus", "T007": "Bacterium",
    "T013": "Fish", "T012": "Bird", "T015": "Mammal", "T005": "Virus",
    "T014": "Reptile", "T011": "Amphibian", "T194": "Archaeon", "T001": "Organism",
    "T008": "Animal", "T016": "Human", "T010": "Vertebrate",
    "T023": "Body Part, Organ, or Organ Component", "T029": "Body Location or Region",
    "T030": "Body Space or Junction", "T026": "Cell Component", "T024": "Tissue",
    "T025": "Cell", "T032": "Organism Attribute", "T022": "Body System",
    "T018": "Embryonic Structure", "T017": "Anatomical Structure",
    "T021": "Fully Formed Anatomical Structure", "T019": "Congenital Abnormality",
    "T190": "Anatomical Abnormality", "T020": "Acquired Abnormality",
    "T028": "Gene or Genome", "T086": "Nucleotide Sequence", "T087": "Amino Acid Sequence",
    "T085": "Molecular Sequence", "T088": "Carbohydrate Sequence",
    "T047": "Disease or Syndrome", "T037": "Injury or Poisoning", "T191": "Neoplastic Process",
    "T048": "Mental or Behavioral Dysfunction", "T050": "Experimental Model of Disease",
    "T033": "Finding", "T184": "Sign or Symptom", "T034": "Laboratory or Test Result",
    "T046": "Pathologic Function", "T044": "Molecular Function", "T043": "Cell Function",
    "T049": "Cell or Molecular Dysfunction", "T040": "Organism Function",
    "T045": "Genetic Function", "T042": "Organ or Tissue Function",
    "T039": "Physiologic Function", "T038": "Biologic Function", "T041": "Mental Process",
    "T067": "Phenomenon or Process", "T070": "Natural Phenomenon or Process",
    "T068": "Human-caused Phenomenon or Process", "T069": "Environmental Effect of Humans",
    "T109": "Organic Chemical", "T167": "Substance", "T197": "Inorganic Chemical",
    "T131": "Hazardous or Poisonous Substance", "T196": "Element, Ion, or Isotope",
    "T104": "Chemical Viewed Structurally", "T120": "Chemical Viewed Functionally",
    "T103": "Chemical", "T200": "Clinical Drug", "T121": "Pharmacologic Substance",
    "T116": "Amino Acid, Peptide, or Protein", "T123": "Biologically Active Substance",
    "T129": "Immunologic Factor", "T126": "Enzyme", "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T195": "Antibiotic", "T192": "Receptor", "T125": "Hormone", "T127": "Vitamin",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
    "T061": "Therapeutic or Preventive Procedure", "T060": "Diagnostic Procedure",
    "T059": "Laboratory Procedure", "T063": "Molecular Biology Research Technique",
    "T074": "Medical Device", "T122": "Biomedical or Dental Material",
    "T203": "Drug Delivery Device", "T075": "Research Device", "T073": "Manufactured Object",
    "T072": "Physical Object", "T058": "Health Care Activity", "T065": "Educational Activity",
    "T062": "Research Activity", "T057": "Occupational Activity", "T054": "Social Behavior",
    "T055": "Individual Behavior", "T056": "Daily or Recreational Activity",
    "T052": "Activity", "T051": "Event", "T066": "Machine Activity", "T053": "Behavior",
    "T201": "Clinical Attribute", "T170": "Intellectual Product", "T081": "Quantitative Concept",
    "T168": "Food", "T080": "Qualitative Concept", "T169": "Functional Concept",
    "T082": "Spatial Concept", "T079": "Temporal Concept", "T078": "Idea or Concept",
    "T185": "Classification", "T031": "Body Substance", "T171": "Language",
    "T077": "Conceptual Entity", "T102": "Group Attribute", "T071": "Entity",
    "T097": "Professional or Occupational Group", "T098": "Population Group",
    "T093": "Health Care Related Organization", "T091": "Biomedical Occupation or Discipline",
    "T090": "Occupation or Discipline", "T064": "Governmental or Regulatory Activity",
    "T089": "Regulation or Law", "T092": "Organization", "T099": "Family Group",
    "T101": "Patient or Disabled Group", "T100": "Age Group",
    "T095": "Self-help or Relief Organization", "T096": "Group", "T094": "Professional Society",
    "T083": "Geographic Area"
}


class ExtractedEntity(BaseModel):
    """A single extracted entity.

    - name: canonical name or identifier
    - mention: optional surface text from the input that triggered the extraction
    """
    name: str = Field(..., description="Canonical name or identifier of the entity")
    mention: Optional[str] = Field(default=None, description="Surface mention in the input text")
    semantic_type: str = Field(..., description="UMLS TUI code for the semantic type")


# ---------------------------------------------------------------------------
# Output schemas for each prompt group (1..15) based on UMLS Semantic Groups
# Each schema maps the TUI codes used in the corresponding prompt group
# to a list of ExtractedEntity objects. Fields default to empty lists.
# ---------------------------------------------------------------------------


class Group1ActivitiesOutput(BaseModel):
    """Output schema for Group 1: Activities & Behaviors."""
    T052: List[ExtractedEntity] = Field(default_factory=list, description="Activity")
    T053: List[ExtractedEntity] = Field(default_factory=list, description="Behavior")
    T054: List[ExtractedEntity] = Field(default_factory=list, description="Social Behavior")
    T055: List[ExtractedEntity] = Field(default_factory=list, description="Individual Behavior")
    T056: List[ExtractedEntity] = Field(default_factory=list, description="Daily or Recreational Activity")
    T057: List[ExtractedEntity] = Field(default_factory=list, description="Occupational Activity")
    T058: List[ExtractedEntity] = Field(default_factory=list, description="Health Care Activity")
    T062: List[ExtractedEntity] = Field(default_factory=list, description="Research Activity")
    T064: List[ExtractedEntity] = Field(default_factory=list, description="Governmental or Regulatory Activity")
    T065: List[ExtractedEntity] = Field(default_factory=list, description="Educational Activity")
    T066: List[ExtractedEntity] = Field(default_factory=list, description="Machine Activity")


class Group2AnatomyOutput(BaseModel):
    """Output schema for Group 2: Anatomy."""
    T017: List[ExtractedEntity] = Field(default_factory=list, description="Anatomical Structure")
    T018: List[ExtractedEntity] = Field(default_factory=list, description="Embryonic Structure")
    T021: List[ExtractedEntity] = Field(default_factory=list, description="Fully Formed Anatomical Structure")
    T022: List[ExtractedEntity] = Field(default_factory=list, description="Body System")
    T023: List[ExtractedEntity] = Field(default_factory=list, description="Body Part, Organ, or Organ Component")
    T024: List[ExtractedEntity] = Field(default_factory=list, description="Tissue")
    T025: List[ExtractedEntity] = Field(default_factory=list, description="Cell")
    T026: List[ExtractedEntity] = Field(default_factory=list, description="Cell Component")
    T029: List[ExtractedEntity] = Field(default_factory=list, description="Body Location or Region")
    T030: List[ExtractedEntity] = Field(default_factory=list, description="Body Space or Junction")


class Group3ChemicalsOutput(BaseModel):
    """Output schema for Group 3: Chemicals & Drugs."""
    T103: List[ExtractedEntity] = Field(default_factory=list, description="Chemical")
    T104: List[ExtractedEntity] = Field(default_factory=list, description="Chemical Viewed Structurally")
    T109: List[ExtractedEntity] = Field(default_factory=list, description="Organic Chemical")
    T114: List[ExtractedEntity] = Field(default_factory=list, description="Nucleic Acid, Nucleoside, or Nucleotide")
    T116: List[ExtractedEntity] = Field(default_factory=list, description="Amino Acid, Peptide, or Protein")
    T120: List[ExtractedEntity] = Field(default_factory=list, description="Chemical Viewed Functionally")
    T121: List[ExtractedEntity] = Field(default_factory=list, description="Pharmacologic Substance")
    T122: List[ExtractedEntity] = Field(default_factory=list, description="Biomedical or Dental Material")
    T123: List[ExtractedEntity] = Field(default_factory=list, description="Biologically Active Substance")
    T125: List[ExtractedEntity] = Field(default_factory=list, description="Hormone")
    T126: List[ExtractedEntity] = Field(default_factory=list, description="Enzyme")
    T127: List[ExtractedEntity] = Field(default_factory=list, description="Vitamin")
    T129: List[ExtractedEntity] = Field(default_factory=list, description="Immunologic Factor")
    T130: List[ExtractedEntity] = Field(default_factory=list, description="Indicator, Reagent, or Diagnostic Aid")
    T131: List[ExtractedEntity] = Field(default_factory=list, description="Hazardous or Poisonous Substance")
    T167: List[ExtractedEntity] = Field(default_factory=list, description="Substance")
    T192: List[ExtractedEntity] = Field(default_factory=list, description="Receptor")
    T195: List[ExtractedEntity] = Field(default_factory=list, description="Antibiotic")
    T196: List[ExtractedEntity] = Field(default_factory=list, description="Element, Ion, or Isotope")
    T197: List[ExtractedEntity] = Field(default_factory=list, description="Inorganic Chemical")
    T200: List[ExtractedEntity] = Field(default_factory=list, description="Clinical Drug")


class Group4ConceptsOutput(BaseModel):
    """Output schema for Group 4: Concepts & Ideas."""
    T031: List[ExtractedEntity] = Field(default_factory=list, description="Body Substance")
    T071: List[ExtractedEntity] = Field(default_factory=list, description="Entity")
    T077: List[ExtractedEntity] = Field(default_factory=list, description="Conceptual Entity")
    T078: List[ExtractedEntity] = Field(default_factory=list, description="Idea or Concept")
    T079: List[ExtractedEntity] = Field(default_factory=list, description="Temporal Concept")
    T080: List[ExtractedEntity] = Field(default_factory=list, description="Qualitative Concept")
    T081: List[ExtractedEntity] = Field(default_factory=list, description="Quantitative Concept")
    T082: List[ExtractedEntity] = Field(default_factory=list, description="Spatial Concept")
    T089: List[ExtractedEntity] = Field(default_factory=list, description="Regulation or Law")
    T102: List[ExtractedEntity] = Field(default_factory=list, description="Group Attribute")
    T168: List[ExtractedEntity] = Field(default_factory=list, description="Food")
    T169: List[ExtractedEntity] = Field(default_factory=list, description="Functional Concept")
    T170: List[ExtractedEntity] = Field(default_factory=list, description="Intellectual Product")
    T171: List[ExtractedEntity] = Field(default_factory=list, description="Language")
    T185: List[ExtractedEntity] = Field(default_factory=list, description="Classification")
    T201: List[ExtractedEntity] = Field(default_factory=list, description="Clinical Attribute")


class Group5DevicesOutput(BaseModel):
    """Output schema for Group 5: Devices."""
    T074: List[ExtractedEntity] = Field(default_factory=list, description="Medical Device")
    T075: List[ExtractedEntity] = Field(default_factory=list, description="Research Device")
    T203: List[ExtractedEntity] = Field(default_factory=list, description="Drug Delivery Device")


class Group6DisordersOutput(BaseModel):
    """Output schema for Group 6: Disorders."""
    T019: List[ExtractedEntity] = Field(default_factory=list, description="Congenital Abnormality")
    T020: List[ExtractedEntity] = Field(default_factory=list, description="Acquired Abnormality")
    T033: List[ExtractedEntity] = Field(default_factory=list, description="Finding")
    T034: List[ExtractedEntity] = Field(default_factory=list, description="Laboratory or Test Result")
    T037: List[ExtractedEntity] = Field(default_factory=list, description="Injury or Poisoning")
    T047: List[ExtractedEntity] = Field(default_factory=list, description="Disease or Syndrome")
    T048: List[ExtractedEntity] = Field(default_factory=list, description="Mental or Behavioral Dysfunction")
    T050: List[ExtractedEntity] = Field(default_factory=list, description="Experimental Model of Disease")
    T184: List[ExtractedEntity] = Field(default_factory=list, description="Sign or Symptom")
    T190: List[ExtractedEntity] = Field(default_factory=list, description="Anatomical Abnormality")
    T191: List[ExtractedEntity] = Field(default_factory=list, description="Neoplastic Process")


class Group7GenesOutput(BaseModel):
    """Output schema for Group 7: Genes & Molecular Sequences."""
    T028: List[ExtractedEntity] = Field(default_factory=list, description="Gene or Genome")
    T085: List[ExtractedEntity] = Field(default_factory=list, description="Molecular Sequence")
    T086: List[ExtractedEntity] = Field(default_factory=list, description="Nucleotide Sequence")
    T087: List[ExtractedEntity] = Field(default_factory=list, description="Amino Acid Sequence")
    T088: List[ExtractedEntity] = Field(default_factory=list, description="Carbohydrate Sequence")


class Group8GeographyOutput(BaseModel):
    """Output schema for Group 8: Geographic Areas."""
    T083: List[ExtractedEntity] = Field(default_factory=list, description="Geographic Area")


class Group9LivingBeingsOutput(BaseModel):
    """Output schema for Group 9: Living Beings."""
    T001: List[ExtractedEntity] = Field(default_factory=list, description="Organism")
    T002: List[ExtractedEntity] = Field(default_factory=list, description="Plant")
    T004: List[ExtractedEntity] = Field(default_factory=list, description="Fungus")
    T005: List[ExtractedEntity] = Field(default_factory=list, description="Virus")
    T007: List[ExtractedEntity] = Field(default_factory=list, description="Bacterium")
    T008: List[ExtractedEntity] = Field(default_factory=list, description="Animal")
    T010: List[ExtractedEntity] = Field(default_factory=list, description="Vertebrate")
    T011: List[ExtractedEntity] = Field(default_factory=list, description="Amphibian")
    T012: List[ExtractedEntity] = Field(default_factory=list, description="Bird")
    T013: List[ExtractedEntity] = Field(default_factory=list, description="Fish")
    T014: List[ExtractedEntity] = Field(default_factory=list, description="Reptile")
    T015: List[ExtractedEntity] = Field(default_factory=list, description="Mammal")
    T016: List[ExtractedEntity] = Field(default_factory=list, description="Human")
    T194: List[ExtractedEntity] = Field(default_factory=list, description="Archaeon")
    T204: List[ExtractedEntity] = Field(default_factory=list, description="Eukaryote")


class Group10ObjectsOutput(BaseModel):
    """Output schema for Group 10: Objects."""
    T072: List[ExtractedEntity] = Field(default_factory=list, description="Physical Object")
    T073: List[ExtractedEntity] = Field(default_factory=list, description="Manufactured Object")


class Group11OccupationsOutput(BaseModel):
    """Output schema for Group 11: Occupations."""
    T090: List[ExtractedEntity] = Field(default_factory=list, description="Occupation or Discipline")
    T091: List[ExtractedEntity] = Field(default_factory=list, description="Biomedical Occupation or Discipline")
    T097: List[ExtractedEntity] = Field(default_factory=list, description="Professional or Occupational Group")


class Group12OrganizationsOutput(BaseModel):
    """Output schema for Group 12: Organizations."""
    T032: List[ExtractedEntity] = Field(default_factory=list, description="Organism Attribute")
    T092: List[ExtractedEntity] = Field(default_factory=list, description="Organization")
    T093: List[ExtractedEntity] = Field(default_factory=list, description="Health Care Related Organization")
    T094: List[ExtractedEntity] = Field(default_factory=list, description="Professional Society")
    T095: List[ExtractedEntity] = Field(default_factory=list, description="Self-help or Relief Organization")
    T096: List[ExtractedEntity] = Field(default_factory=list, description="Group")
    T098: List[ExtractedEntity] = Field(default_factory=list, description="Population Group")
    T099: List[ExtractedEntity] = Field(default_factory=list, description="Family Group")
    T100: List[ExtractedEntity] = Field(default_factory=list, description="Age Group")
    T101: List[ExtractedEntity] = Field(default_factory=list, description="Patient or Disabled Group")


class Group13PhenomenaOutput(BaseModel):
    """Output schema for Group 13: Phenomena."""
    T051: List[ExtractedEntity] = Field(default_factory=list, description="Event")
    T067: List[ExtractedEntity] = Field(default_factory=list, description="Phenomenon or Process")
    T068: List[ExtractedEntity] = Field(default_factory=list, description="Human-caused Phenomenon or Process")
    T069: List[ExtractedEntity] = Field(default_factory=list, description="Environmental Effect of Humans")
    T070: List[ExtractedEntity] = Field(default_factory=list, description="Natural Phenomenon or Process")


class Group14PhysiologyOutput(BaseModel):
    """Output schema for Group 14: Physiology."""
    T038: List[ExtractedEntity] = Field(default_factory=list, description="Biologic Function")
    T039: List[ExtractedEntity] = Field(default_factory=list, description="Physiologic Function")
    T040: List[ExtractedEntity] = Field(default_factory=list, description="Organism Function")
    T041: List[ExtractedEntity] = Field(default_factory=list, description="Mental Process")
    T042: List[ExtractedEntity] = Field(default_factory=list, description="Organ or Tissue Function")
    T043: List[ExtractedEntity] = Field(default_factory=list, description="Cell Function")
    T044: List[ExtractedEntity] = Field(default_factory=list, description="Molecular Function")
    T045: List[ExtractedEntity] = Field(default_factory=list, description="Genetic Function")
    T046: List[ExtractedEntity] = Field(default_factory=list, description="Pathologic Function")
    T049: List[ExtractedEntity] = Field(default_factory=list, description="Cell or Molecular Dysfunction")


class Group15ProceduresOutput(BaseModel):
    """Output schema for Group 15: Procedures."""
    T059: List[ExtractedEntity] = Field(default_factory=list, description="Laboratory Procedure")
    T060: List[ExtractedEntity] = Field(default_factory=list, description="Diagnostic Procedure")
    T061: List[ExtractedEntity] = Field(default_factory=list, description="Therapeutic or Preventive Procedure")
    T063: List[ExtractedEntity] = Field(default_factory=list, description="Molecular Biology Research Technique")


__all__ = [
    'ExtractedEntity',
    'TUI_TO_SEMTYPE',
    'Group1ActivitiesOutput',
    'Group2AnatomyOutput',
    'Group3ChemicalsOutput',
    'Group4ConceptsOutput',
    'Group5DevicesOutput',
    'Group6DisordersOutput',
    'Group7GenesOutput',
    'Group8GeographyOutput',
    'Group9LivingBeingsOutput',
    'Group10ObjectsOutput',
    'Group11OccupationsOutput',
    'Group12OrganizationsOutput',
    'Group13PhenomenaOutput',
    'Group14PhysiologyOutput',
    'Group15ProceduresOutput',
]
