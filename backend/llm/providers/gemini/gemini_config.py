from typing import Dict, Any, Optional, List
from pydantic import Field
from ...base.llm_config import BaseLLMConfig

class GeminiConfig(BaseLLMConfig):
    model_name: str = Field(default="gemini-2.0-flash")
    embedding_model: str = Field(default="text-embedding-004")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    
    def get_generation_config(self) -> Dict[str, Any]:
        config = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.top_k is not None:
            config["top_k"] = self.top_k
        return config