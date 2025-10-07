from pydantic import BaseModel, Field
from typing import Dict, Any, List
import uuid

class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "Entity"
    properties: Dict[str, Any] = Field(default_factory=dict)

class Edge(BaseModel):
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class Graph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
