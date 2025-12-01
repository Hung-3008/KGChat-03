from typing import Dict, Any, Optional
from pydantic import Field
from ...base.llm_config import BaseLLMConfig
from ...utils.config_builder import build_generation_config


class OllamaConfig(BaseLLMConfig):
    model_name: str = Field(default="llama3.1:8b")
    embedding_model: str = Field(default="llama3.1:8b")
    host: str = Field(default="http://localhost:11434")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)

    def get_generation_config(self) -> Dict[str, Any]:
        return build_generation_config(self)