from typing import Dict, Any, Optional
from pydantic import Field
from ...base.llm_config import BaseLLMConfig

class OpenAIConfig(BaseLLMConfig):
    model_name: str = Field(default="gpt-4")
    embedding_model: str = Field(default="text-embedding-ada-002")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)

    def get_generation_config(self) -> Dict[str, Any]:
        config = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        return config
