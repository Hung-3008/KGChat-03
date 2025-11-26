from pydantic import BaseModel, Field
from typing import List, Dict
import json

class BrokenModel(BaseModel):
    items: List[str] = Field(default_factory=list)

class FixedModel(BaseModel):
    items: list[str] = Field(default_factory=list)

print("Testing BrokenModel...")
try:
    print(json.dumps(BrokenModel.model_json_schema(), indent=2))
except Exception as e:
    print(f"BrokenModel failed: {e}")

print("\nTesting FixedModel...")
try:
    print(json.dumps(FixedModel.model_json_schema(), indent=2))
except Exception as e:
    print(f"FixedModel failed: {e}")
