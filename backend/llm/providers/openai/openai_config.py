from typing import Dict, Any, Optional
from pydantic import Field
from ...base.llm_config import BaseLLMConfig

class OpenAIConfig(BaseLLMConfig):
    # Bạn có thể đổi model theo nhu cầu: "gpt-4o-mini", "gpt-4.1-mini", v.v.
    model_name: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Tham số sampling/phạt tương đương để map sang OpenAI
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    def get_generation_config(self) -> Dict[str, Any]:
        """
        Trả về dict dùng cho OpenAI Chat Completions.
        Những key None sẽ không được set để giữ API payload gọn gàng.
        """
        cfg: Dict[str, Any] = {}

        if self.max_tokens is not None:
            cfg["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self.top_p is not None:
            cfg["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            cfg["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            cfg["presence_penalty"] = self.presence_penalty

        return cfg