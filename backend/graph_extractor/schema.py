from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Optimized schemas matching reduced semantic types in prompts

class Activity(BaseModel):
    """Reduced activity types"""
    Health_Care_Activity: List[str] = Field(default_factory=list)
    Laboratory_Procedure: List[str] = Field(default_factory=list)
    Diagnostic_Procedure: List[str] = Field(default_factory=list)
    Therapeutic_or_Preventive_Procedure: List[str] = Field(default_factory=list)
    Research_Activity: List[str] = Field(default_factory=list)
    Behavior: List[str] = Field(default_factory=list)


class Phenomenon(BaseModel):
    """Reduced phenomenon types"""
    Disease_or_Syndrome: List[str] = Field(default_factory=list)
    Neoplastic_Process: List[str] = Field(default_factory=list)
    Injury_or_Poisoning: List[str] = Field(default_factory=list)
    Pathologic_Function: List[str] = Field(default_factory=list)
    Sign_or_Symptom: List[str] = Field(default_factory=list)


class PhysicalObject(BaseModel):
    """Reduced physical object types"""
    Body_Part_Organ_or_Organ_Component: List[str] = Field(default_factory=list)
    Tissue: List[str] = Field(default_factory=list)
    Cell: List[str] = Field(default_factory=list)
    Gene_or_Genome: List[str] = Field(default_factory=list)
    Anatomical_Abnormality: List[str] = Field(default_factory=list)
    Clinical_Drug: List[str] = Field(default_factory=list)
    Pharmacologic_Substance: List[str] = Field(default_factory=list)
    Medical_Device: List[str] = Field(default_factory=list)
    Body_Substance: List[str] = Field(default_factory=list)
    Organism: List[str] = Field(default_factory=list)


class ConceptualEntity(BaseModel):
    """Reduced conceptual entity types"""
    Finding: List[str] = Field(default_factory=list)
    Laboratory_or_Test_Result: List[str] = Field(default_factory=list)
    Anatomical_Concept: List[str] = Field(default_factory=list)
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
