import openai
from typing import Optional, Dict, Any
import logging
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .openai_config import OpenAIConfig

logger = logging.getLogger(__name__)

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.config: OpenAIConfig = config
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.config.api_key:
            raise AuthenticationError("OpenAI API key is required")

        openai.api_key = self.config.api_key
        logger.info(f"Initialized OpenAI client with model: {self.config.model_name}")

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )

            choice = response.choices[0]
            message = choice.message['content']

            metadata = {
                "model": self.config.model_name,
                "finish_reason": choice.get('finish_reason', None),
                "usage": response.get("usage", None)
            }

            return LLMResponse(message=message, metadata=metadata)

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise