from typing import Dict, Any, Optional
from pydantic import Field
from ...base.llm_config import BaseLLMConfig
from ...utils.config_builder import build_generation_config


class OpenAIConfig(BaseLLMConfig):
    model_name: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    def get_generation_config(self) -> Dict[str, Any]:
        return build_generation_config(
            self,
            include=[
                "max_tokens",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ],
        )