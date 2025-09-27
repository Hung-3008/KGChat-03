rom typing import Optional, Dict, Any
import logging

from openai import OpenAI
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .openai_config import OpenAIConfig

logger = logging.getLogger(__name__)

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.config: OpenAIConfig = config
        self._client: Optional[OpenAI] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.config.api_key:
            raise AuthenticationError("OpenAI API key is required")

        self._client = OpenAI(api_key=self.config.api_key)
        logger.info(f"Initialized OpenAI client with model: {self.config.model_name}")

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized")

        try:
            gen_cfg: Dict[str, Any] = self.config.get_generation_config()
            if kwargs:
                gen_cfg.update({k: v for k, v in kwargs.items() if v is not None})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            resp = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                **gen_cfg,
            )

            choice = resp.choices[0]
            message_text = choice.message.content or "No response generated"

            metadata = {
                "model": resp.model,
                "finish_reason": choice.finish_reason,
                "usage": getattr(resp, "usage", None),
            }

            return LLMResponse(message=message_text, metadata=metadata)

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise