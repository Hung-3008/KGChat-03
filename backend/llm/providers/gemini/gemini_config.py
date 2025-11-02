from typing import Dict, Any, Optional
from pydantic import Field
from ...base.llm_config import BaseLLMConfig
from ...utils.config_builder import build_generation_config


class GeminiConfig(BaseLLMConfig):
    model_name: str = Field(default="gemini-1.5-flash")
    embedding_model: str = Field(default="models/text-embedding-004")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    response_mime_type: Optional[str] = None

    def get_generation_config(self) -> Dict[str, Any]:
        cfg = build_generation_config(self, field_map={"max_tokens": "max_output_tokens"})
        if self.response_mime_type:
            cfg["response_mime_type"] = self.response_mime_type
        return cfg
