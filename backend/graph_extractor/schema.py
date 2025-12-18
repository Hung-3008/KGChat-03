from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Optimized schemas matching reduced semantic types in prompts

class Activity(BaseModel):
    """Reduced from 15 to 9 activity types"""
    Health_Care_Activity: List[str] = Field(default_factory=list)
    Laboratory_Procedure: List[str] = Field(default_factory=list)
    Diagnostic_Procedure: List[str] = Field(default_factory=list)
    Therapeutic_or_Preventive_Procedure: List[str] = Field(default_factory=list)
    Research_Activity: List[str] = Field(default_factory=list)
    Molecular_Biology_Research_Technique: List[str] = Field(default_factory=list)
    Governmental_or_Regulatory_Activity: List[str] = Field(default_factory=list)
    Behavior: List[str] = Field(default_factory=list)
    Activity: List[str] = Field(default_factory=list)


class Phenomenon(BaseModel):
    """Reduced from 19 to 11 types"""
    Disease_or_Syndrome: List[str] = Field(default_factory=list)
    Neoplastic_Process: List[str] = Field(default_factory=list)
    Injury_or_Poisoning: List[str] = Field(default_factory=list)
    Pathologic_Function: List[str] = Field(default_factory=list)
    Mental_or_Behavioral_Dysfunction: List[str] = Field(default_factory=list)
    Sign_or_Symptom: List[str] = Field(default_factory=list)
    Physiologic_Function: List[str] = Field(default_factory=list)
    Organ_or_Tissue_Function: List[str] = Field(default_factory=list)
    Cell_or_Molecular_Dysfunction: List[str] = Field(default_factory=list)
    Biologic_Function: List[str] = Field(default_factory=list)
    Mental_Process: List[str] = Field(default_factory=list)


class PhysicalObject(BaseModel):
    """Reduced from 55 to 16 types"""
    Body_Part_Organ_or_Organ_Component: List[str] = Field(default_factory=list)
    Tissue: List[str] = Field(default_factory=list)
    Cell: List[str] = Field(default_factory=list)
    Gene_or_Genome: List[str] = Field(default_factory=list)
    Anatomical_Abnormality: List[str] = Field(default_factory=list)
    Congenital_Abnormality: List[str] = Field(default_factory=list)
    Acquired_Abnormality: List[str] = Field(default_factory=list)
    Clinical_Drug: List[str] = Field(default_factory=list)
    Pharmacologic_Substance: List[str] = Field(default_factory=list)
    Antibiotic: List[str] = Field(default_factory=list)
    Hormone: List[str] = Field(default_factory=list)
    Enzyme: List[str] = Field(default_factory=list)
    Vitamin: List[str] = Field(default_factory=list)
    Medical_Device: List[str] = Field(default_factory=list)
    Body_Substance: List[str] = Field(default_factory=list)
    Organism: List[str] = Field(default_factory=list)


class ConceptualEntity(BaseModel):
    """Reduced from 38 to 8 types"""
    Finding: List[str] = Field(default_factory=list)
    Sign_or_Symptom: List[str] = Field(default_factory=list)
    Laboratory_or_Test_Result: List[str] = Field(default_factory=list)
    Clinical_Attribute: List[str] = Field(default_factory=list)
    Body_Location_or_Region: List[str] = Field(default_factory=list)
    Body_System: List[str] = Field(default_factory=list)
    Quantitative_Concept: List[str] = Field(default_factory=list)
    Temporal_Concept: List[str] = Field(default_factory=list)


class Entity(BaseModel):
    name: str 
    semantic_type: str
    mention: str
    context_left: str = ""
    context_right: str = ""

class ValidatedEntity(BaseModel):
    entities: List[Entity]


class Edge(BaseModel):
    source: str
    target: str
    relation: str
    evidence: str

class ExtractedEdges(BaseModel):
    edges: List[Edge]

__all__ = [
    "Activity", "Phenomenon", "PhysicalObject", "ConceptualEntity",
    "Entity", "ValidatedEntity", "Edge", "ExtractedEdges"
]
