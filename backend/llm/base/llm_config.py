from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class BaseLLMConfig(BaseModel):
    model_name: str
    max_tokens: Optional[int] = Field(default=8192, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    api_key: Optional[str] = None
    
    class Config:
        extra = "allow"