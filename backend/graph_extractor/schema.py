from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Activity(BaseModel):
    Activity: List[str] = Field(
        default_factory=list, 
        description="List of general activities mentioned in the clinical text"
    )
    Behavior: List[str] = Field(
        default_factory=list, 
        description="List of behaviors mentioned in the clinical text"
    )
    Social_Behavior: List[str] = Field(
        default_factory=list, 
        description="List of social behaviors mentioned in the clinical text"
    )
    Individual_Behavior: List[str] = Field(
        default_factory=list, 
        description="List of individual behaviors mentioned in the clinical text"
    )
    Daily_or_Recreational_Activity: List[str] = Field(
        default_factory=list, 
        description="List of daily or recreational activities mentioned in the clinical text"
    )
    Occupational_Activity: List[str] = Field(
        default_factory=list, 
        description="List of occupational activities mentioned in the clinical text"
    )
    Health_Care_Activity: List[str] = Field(
        default_factory=list, 
        description="List of health care activities mentioned in the clinical text"
    )
    Laboratory_Procedure: List[str] = Field(
        default_factory=list, 
        description="List of laboratory procedures mentioned in the clinical text"
    )
    Diagnostic_Procedure: List[str] = Field(
        default_factory=list, 
        description="List of diagnostic procedures mentioned in the clinical text"
    )
    Therapeutic_or_Preventive_Procedure: List[str] = Field(
        default_factory=list, 
        description="List of therapeutic or preventive procedures mentioned in the clinical text"
    )
    Research_Activity: List[str] = Field(
        default_factory=list, 
        description="List of research activities mentioned in the clinical text"
    )
    Molecular_Biology_Research_Technique: List[str] = Field(
        default_factory=list, 
        description="List of molecular biology research techniques mentioned in the clinical text"
    )
    Governmental_or_Regulatory_Activity: List[str] = Field(
        default_factory=list, 
        description="List of governmental or regulatory activities mentioned in the clinical text"
    )
    Educational_Activity: List[str] = Field(
        default_factory=list, 
        description="List of educational activities mentioned in the clinical text"
    )
    Machine_Activity: List[str] = Field(
        default_factory=list, 
        description="List of machine activities mentioned in the clinical text"
    )


class Phenomenon(BaseModel):
    Phenomenon_or_Process: List[str] = Field(
        default_factory=list,
        description="General phenomena or processes mentioned in the clinical text"
    )
    Injury_or_Poisoning: List[str] = Field(
        default_factory=list,
        description="Injuries or poisonings mentioned in the clinical text"
    )
    Human_caused_Phenomenon_or_Process: List[str] = Field(
        default_factory=list,
        description="Phenomena or processes caused by humans mentioned in the clinical text"
    )
    Environmental_Effect_of_Humans: List[str] = Field(
        default_factory=list,
        description="Environmental effects caused by human activities mentioned in the clinical text"
    )
    Natural_Phenomenon_or_Process: List[str] = Field(
        default_factory=list,
        description="Natural phenomena or processes mentioned in the clinical text"
    )
    Biologic_Function: List[str] = Field(
        default_factory=list,
        description="Biological functions mentioned in the clinical text"
    )
    Physiologic_Function: List[str] = Field(
        default_factory=list,
        description="Physiological functions mentioned in the clinical text"
    )
    Organism_Function: List[str] = Field(
        default_factory=list,
        description="Functions of organisms mentioned in the clinical text"
    )
    Mental_Process: List[str] = Field(
        default_factory=list,
        description="Mental processes mentioned in the clinical text"
    )
    Organ_or_Tissue_Function: List[str] = Field(
        default_factory=list,
        description="Functions of organs or tissues mentioned in the clinical text"
    )
    Cell_Function: List[str] = Field(
        default_factory=list,
        description="Cellular functions mentioned in the clinical text"
    )
    Molecular_Function: List[str] = Field(
        default_factory=list,
        description="Molecular functions mentioned in the clinical text"
    )
    Genetic_Function: List[str] = Field(
        default_factory=list,
        description="Genetic functions mentioned in the clinical text"
    )
    Cell_or_Molecular_Dysfunction: List[str] = Field(
        default_factory=list,
        description="Dysfunctions at cellular or molecular level mentioned in the clinical text"
    )
    Pathologic_Function: List[str] = Field(
        default_factory=list,
        description="Pathological functions mentioned in the clinical text"
    )
    Disease_or_Syndrome: List[str] = Field(
        default_factory=list,
        description="Diseases or syndromes mentioned in the clinical text"
    )
    Mental_or_Behavioral_Dysfunction: List[str] = Field(
        default_factory=list,
        description="Mental or behavioral dysfunctions mentioned in the clinical text"
    )
    Neoplastic_Process: List[str] = Field(
        default_factory=list,
        description="Neoplastic processes (tumor/cancer-related) mentioned in the clinical text"
    )
    Experimental_Model_of_Disease: List[str] = Field(
        default_factory=list,
        description="Experimental models of diseases mentioned in the clinical text"
    )


class PhysicalObject(BaseModel):
    Physical_Object: List[str] = Field(
        default_factory=list,
        description="General physical objects mentioned in the clinical text"
    )
    Organism: List[str] = Field(
        default_factory=list,
        description="Organisms mentioned in the clinical text"
    )
    Virus: List[str] = Field(
        default_factory=list,
        description="Viruses mentioned in the clinical text"
    )
    Bacterium: List[str] = Field(
        default_factory=list,
        description="Bacteria mentioned in the clinical text"
    )
    Archaeon: List[str] = Field(
        default_factory=list,
        description="Archaeons mentioned in the clinical text"
    )
    Eukaryote: List[str] = Field(
        default_factory=list,
        description="Eukaryotes mentioned in the clinical text"
    )
    Plant: List[str] = Field(
        default_factory=list,
        description="Plants mentioned in the clinical text"
    )
    Fungus: List[str] = Field(
        default_factory=list,
        description="Fungi mentioned in the clinical text"
    )
    Animal: List[str] = Field(
        default_factory=list,
        description="Animals mentioned in the clinical text"
    )
    Vertebrate: List[str] = Field(
        default_factory=list,
        description="Vertebrates mentioned in the clinical text"
    )
    Amphibian: List[str] = Field(
        default_factory=list,
        description="Amphibians mentioned in the clinical text"
    )
    Bird: List[str] = Field(
        default_factory=list,
        description="Birds mentioned in the clinical text"
    )
    Fish: List[str] = Field(
        default_factory=list,
        description="Fish mentioned in the clinical text"
    )
    Reptile: List[str] = Field(
        default_factory=list,
        description="Reptiles mentioned in the clinical text"
    )
    Mammal: List[str] = Field(
        default_factory=list,
        description="Mammals mentioned in the clinical text"
    )
    Human: List[str] = Field(
        default_factory=list,
        description="Humans mentioned in the clinical text"
    )
    Anatomical_Structure: List[str] = Field(
        default_factory=list,
        description="Anatomical structures mentioned in the clinical text"
    )
    Embryonic_Structure: List[str] = Field(
        default_factory=list,
        description="Embryonic structures mentioned in the clinical text"
    )
    Fully_Formed_Anatomical_Structure: List[str] = Field(
        default_factory=list,
        description="Fully formed anatomical structures mentioned in the clinical text"
    )
    Body_Part_Organ_or_Organ_Component: List[str] = Field(
        default_factory=list,
        description="Body parts, organs, or organ components mentioned in the clinical text"
    )
    Tissue: List[str] = Field(
        default_factory=list,
        description="Tissues mentioned in the clinical text"
    )
    Cell: List[str] = Field(
        default_factory=list,
        description="Cells mentioned in the clinical text"
    )
    Cell_Component: List[str] = Field(
        default_factory=list,
        description="Cell components mentioned in the clinical text"
    )
    Gene_or_Genome: List[str] = Field(
        default_factory=list,
        description="Genes or genomes mentioned in the clinical text"
    )
    Anatomical_Abnormality: List[str] = Field(
        default_factory=list,
        description="Anatomical abnormalities mentioned in the clinical text"
    )
    Congenital_Abnormality: List[str] = Field(
        default_factory=list,
        description="Congenital abnormalities mentioned in the clinical text"
    )
    Acquired_Abnormality: List[str] = Field(
        default_factory=list,
        description="Acquired abnormalities mentioned in the clinical text"
    )
    Manufactured_Object: List[str] = Field(
        default_factory=list,
        description="Manufactured objects mentioned in the clinical text"
    )
    Medical_Device: List[str] = Field(
        default_factory=list,
        description="Medical devices mentioned in the clinical text"
    )
    Drug_Delivery_Device: List[str] = Field(
        default_factory=list,
        description="Drug delivery devices mentioned in the clinical text"
    )
    Research_Device: List[str] = Field(
        default_factory=list,
        description="Research devices mentioned in the clinical text"
    )
    Clinical_Drug: List[str] = Field(
        default_factory=list,
        description="Clinical drugs mentioned in the clinical text"
    )
    Substance: List[str] = Field(
        default_factory=list,
        description="Substances mentioned in the clinical text"
    )
    Body_Substance: List[str] = Field(
        default_factory=list,
        description="Body substances mentioned in the clinical text"
    )
    Chemical: List[str] = Field(
        default_factory=list,
        description="Chemicals mentioned in the clinical text"
    )
    Chemical_Viewed_Structurally: List[str] = Field(
        default_factory=list,
        description="Chemicals viewed structurally mentioned in the clinical text"
    )
    Organic_Chemical: List[str] = Field(
        default_factory=list,
        description="Organic chemicals mentioned in the clinical text"
    )
    Nucleic_Acid_Nucleoside_or_Nucleotide: List[str] = Field(
        default_factory=list,
        description="Nucleic acids, nucleosides, or nucleotides mentioned in the clinical text"
    )
    Amino_Acid_Peptide_or_Protein: List[str] = Field(
        default_factory=list,
        description="Amino acids, peptides, or proteins mentioned in the clinical text"
    )
    Element_Ion_or_Isotope: List[str] = Field(
        default_factory=list,
        description="Elements, ions, or isotopes mentioned in the clinical text"
    )
    Inorganic_Chemical: List[str] = Field(
        default_factory=list,
        description="Inorganic chemicals mentioned in the clinical text"
    )
    Chemical_Viewed_Functionally: List[str] = Field(
        default_factory=list,
        description="Chemicals viewed functionally mentioned in the clinical text"
    )
    Pharmacologic_Substance: List[str] = Field(
        default_factory=list,
        description="Pharmacologic substances mentioned in the clinical text"
    )
    Antibiotic: List[str] = Field(
        default_factory=list,
        description="Antibiotics mentioned in the clinical text"
    )
    Biomedical_or_Dental_Material: List[str] = Field(
        default_factory=list,
        description="Biomedical or dental materials mentioned in the clinical text"
    )
    Biologically_Active_Substance: List[str] = Field(
        default_factory=list,
        description="Biologically active substances mentioned in the clinical text"
    )
    Hormone: List[str] = Field(
        default_factory=list,
        description="Hormones mentioned in the clinical text"
    )
    Enzyme: List[str] = Field(
        default_factory=list,
        description="Enzymes mentioned in the clinical text"
    )
    Vitamin: List[str] = Field(
        default_factory=list,
        description="Vitamins mentioned in the clinical text"
    )
    Immunologic_Factor: List[str] = Field(
        default_factory=list,
        description="Immunologic factors mentioned in the clinical text"
    )
    Receptor: List[str] = Field(
        default_factory=list,
        description="Receptors mentioned in the clinical text"
    )
    Indicator_Reagent_or_Diagnostic_Aid: List[str] = Field(
        default_factory=list,
        description="Indicator reagents or diagnostic aids mentioned in the clinical text"
    )
    Hazardous_or_Poisonous_Substance: List[str] = Field(
        default_factory=list,
        description="Hazardous or poisonous substances mentioned in the clinical text"
    )
    Food: List[str] = Field(
        default_factory=list,
        description="Foods mentioned in the clinical text"
    )

class ConceptualEntity(BaseModel):
    Conceptual_Entity: List[str] = Field(
        default_factory=list,
        description="General conceptual entities mentioned in the clinical text"
    )
    Organism_Attribute: List[str] = Field(
        default_factory=list,
        description="Attributes of organisms mentioned in the clinical text"
    )
    Clinical_Attribute: List[str] = Field(
        default_factory=list,
        description="Clinical attributes mentioned in the clinical text"
    )
    Finding: List[str] = Field(
        default_factory=list,
        description="Findings mentioned in the clinical text"
    )
    Laboratory_or_Test_Result: List[str] = Field(
        default_factory=list,
        description="Laboratory or test results mentioned in the clinical text"
    )
    Sign_or_Symptom: List[str] = Field(
        default_factory=list,
        description="Signs or symptoms mentioned in the clinical text"
    )
    Idea_or_Concept: List[str] = Field(
        default_factory=list,
        description="Ideas or concepts mentioned in the clinical text"
    )
    Temporal_Concept: List[str] = Field(
        default_factory=list,
        description="Temporal concepts mentioned in the clinical text"
    )
    Qualitative_Concept: List[str] = Field(
        default_factory=list,
        description="Qualitative concepts mentioned in the clinical text"
    )
    Quantitative_Concept: List[str] = Field(
        default_factory=list,
        description="Quantitative concepts mentioned in the clinical text"
    )
    Spatial_Concept: List[str] = Field(
        default_factory=list,
        description="Spatial concepts mentioned in the clinical text"
    )
    Body_Location_or_Region: List[str] = Field(
        default_factory=list,
        description="Body locations or regions mentioned in the clinical text"
    )
    Body_Space_or_Junction: List[str] = Field(
        default_factory=list,
        description="Body spaces or junctions mentioned in the clinical text"
    )
    Geographic_Area: List[str] = Field(
        default_factory=list,
        description="Geographic areas mentioned in the clinical text"
    )
    Molecular_Sequence: List[str] = Field(
        default_factory=list,
        description="Molecular sequences mentioned in the clinical text"
    )
    Nucleotide_Sequence: List[str] = Field(
        default_factory=list,
        description="Nucleotide sequences mentioned in the clinical text"
    )
    Amino_Acid_Sequence: List[str] = Field(
        default_factory=list,
        description="Amino acid sequences mentioned in the clinical text"
    )
    Carbohydrate_Sequence: List[str] = Field(
        default_factory=list,
        description="Carbohydrate sequences mentioned in the clinical text"
    )
    Functional_Concept: List[str] = Field(
        default_factory=list,
        description="Functional concepts mentioned in the clinical text"
    )
    Body_System: List[str] = Field(
        default_factory=list,
        description="Body systems mentioned in the clinical text"
    )
    Occupation_or_Discipline: List[str] = Field(
        default_factory=list,
        description="Occupations or disciplines mentioned in the clinical text"
    )
    Biomedical_Occupation_or_Discipline: List[str] = Field(
        default_factory=list,
        description="Biomedical occupations or disciplines mentioned in the clinical text"
    )
    Organization: List[str] = Field(
        default_factory=list,
        description="Organizations mentioned in the clinical text"
    )
    Health_Care_Related_Organization: List[str] = Field(
        default_factory=list,
        description="Health care related organizations mentioned in the clinical text"
    )
    Professional_Society: List[str] = Field(
        default_factory=list,
        description="Professional societies mentioned in the clinical text"
    )
    Self_help_or_Relief_Organization: List[str] = Field(
        default_factory=list,
        description="Self-help or relief organizations mentioned in the clinical text"
    )
    Group: List[str] = Field(
        default_factory=list,
        description="Groups mentioned in the clinical text"
    )
    Professional_or_Occupational_Group: List[str] = Field(
        default_factory=list,
        description="Professional or occupational groups mentioned in the clinical text"
    )
    Population_Group: List[str] = Field(
        default_factory=list,
        description="Population groups mentioned in the clinical text"
    )
    Family_Group: List[str] = Field(
        default_factory=list,
        description="Family groups mentioned in the clinical text"
    )
    Age_Group: List[str] = Field(
        default_factory=list,
        description="Age groups mentioned in the clinical text"
    )
    Patient_or_Disabled_Group: List[str] = Field(
        default_factory=list,
        description="Patient or disabled groups mentioned in the clinical text"
    )
    Group_Attribute: List[str] = Field(
        default_factory=list,
        description="Group attributes mentioned in the clinical text"
    )
    Intellectual_Product: List[str] = Field(
        default_factory=list,
        description="Intellectual products mentioned in the clinical text"
    )
    Regulation_or_Law: List[str] = Field(
        default_factory=list,
        description="Regulations or laws mentioned in the clinical text"
    )
    Classification: List[str] = Field(
        default_factory=list,
        description="Classifications mentioned in the clinical text"
    )
    Language: List[str] = Field(
        default_factory=list,
        description="Languages mentioned in the clinical text"
    )


class Entity(BaseModel):
    name: str 
    semantic_type: str
    mention: str

class ValidatedEntity(BaseModel):
    entities: List[Entity]


class Edge(BaseModel):
    source: str
    target: str
    relation: str
    confidence: Optional[float] = None

class ExtractedEdges(BaseModel):
    edges: List[Edge]

__all__ = [

]
