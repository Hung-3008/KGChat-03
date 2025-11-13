from pydantic import BaseModel
from typing import List


class ExtractedEntity(BaseModel):
    name: str
    mention: str


class ExtractionOutput(BaseModel):
    entities: List[ExtractedEntity]
