from pydantic import BaseModel, Field
from typing import Dict, Any
import uuid

class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "Entity"
    properties: Dict[str, Any] = Field(default_factory=dict)
