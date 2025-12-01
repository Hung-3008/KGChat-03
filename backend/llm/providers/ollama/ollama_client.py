import ollama
from typing import Optional, Dict, Any, List

from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .ollama_config import OllamaConfig


class OllamaClient(BaseLLMClient):
    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self.config: OllamaConfig = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.config.host:
            raise AuthenticationError("Ollama host is required")

        self._client = ollama.Client(host=self.config.host)
        self._generation_config = self.config.get_generation_config()

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat(model=self.config.model_name, messages=messages)

        # Normalize response to a string message
        message = ""
        if isinstance(response, dict):
            message = response.get("message", {}).get("content", "")
        else:
            msg = getattr(response, "message", None)
            if isinstance(msg, dict):
                message = msg.get("content", "")
            elif hasattr(msg, "content"):
                message = getattr(msg, "content", "")
            else:
                message = str(response)

        if not message:
            raise RuntimeError("Ollama response invalid or missing content")

        metadata = {
            "model": self.config.model_name,
        }

        return LLMResponse(message=message, metadata=metadata)

    def embed(self, text: str) -> List[float]:
        response = self._client.embed(model=self.config.model_name, input=text)
        if isinstance(response, dict):
            embedding = response.get("embeddings")
        else:
            embedding = getattr(response, "embeddings", None)

        if embedding is None:
            raise ValueError("Embedding not found in Ollama response")

        return embedding



