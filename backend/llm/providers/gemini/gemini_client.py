import google.generativeai as genai
from typing import Optional, Dict, Any, List
import os
import time
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .gemini_config import GeminiConfig


class GeminiClient(BaseLLMClient):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.config: GeminiConfig = config
        self._model = None

        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            raise AuthenticationError("No GEMINI_API_KEY found in environment variables")

        self.current_key_index = 0
        self._initialize_client()

    def _load_api_keys(self) -> List[str]:
        keys = []
        if self.config.api_key:
            keys.append(self.config.api_key)

        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                keys.append(key)
                i += 1
            else:
                break

        legacy_key = os.getenv("GOOGLE_API_KEY")
        if legacy_key and legacy_key not in keys:
            keys.append(legacy_key)

        return keys

    def _rotate_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False

        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        try:
            self._initialize_client()
            return True
        except Exception:
            return False

    def _initialize_client(self) -> None:
        if not self.api_keys:
            raise AuthenticationError("No API keys available")

        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Dict[str, Any]] = None, max_tries: int = 5, **kwargs) -> LLMResponse:
        last_exception = None
        keys_tried = set()

        for attempt in range(max_tries):
            try:
                model = genai.GenerativeModel(model_name=self.config.model_name, system_instruction=system_prompt)

                generation_config_args = self.config.get_generation_config()

                if response_schema:
                    generation_config_args["response_mime_type"] = "application/json"
                    generation_config_args["response_schema"] = response_schema
                elif self.config.response_mime_type:
                    generation_config_args["response_mime_type"] = self.config.response_mime_type

                generation_config = genai.GenerationConfig(**generation_config_args)

                response = model.generate_content(user_prompt, generation_config=generation_config)

                message = response.text if response.parts else ""

                metadata = {
                    "model": self.config.model_name,
                    "finish_reason": getattr(response, 'finish_reason', None),
                    "usage": getattr(response, 'usage_metadata', None)
                }

                return LLMResponse(message=message, metadata=metadata)

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                if any(keyword in error_str for keyword in ["quota", "rate", "limit", "exceeded", "billing"]):
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            continue
                        else:
                            pass
                    else:
                        if attempt < max_tries - 1:
                            time.sleep(2 ** attempt)
                elif "authentication" in error_str or "api_key" in error_str:
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            continue
                    raise e
                else:
                    if attempt < max_tries - 1:
                        time.sleep(1)

        raise last_exception

    def get_api_key_info(self) -> Dict[str, any]:
        return {
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_key_index,
            "current_key_preview": self.api_keys[self.current_key_index][:10] + "..." if self.api_keys else None,
            "all_keys_preview": [key[:10] + "..." for key in self.api_keys]
        }

    def list_api_keys(self) -> None:
        info = self.get_api_key_info()
        # minimal listing; callers can inspect get_api_key_info
        return None

    def embed(self, texts: List[str], model: str = None, max_tries: int = 5) -> List[List[float]]:
        if not texts:
            return []

        embedding_model = model or self.config.embedding_model
        last_exception = None
        keys_tried = set()

        for attempt in range(max_tries):
            try:
                embeddings = []
                for text in texts:
                    result = genai.embed_content(model=embedding_model, content=text, task_type="retrieval_document")
                    embedding = result['embedding']
                    embeddings.append(embedding)

                return embeddings

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                if any(keyword in error_str for keyword in ["quota", "rate", "limit", "exceeded", "billing"]):
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            continue
                        else:
                            pass
                    else:
                        if attempt < max_tries - 1:
                            time.sleep(2 ** attempt)
                elif "authentication" in error_str or "api_key" in error_str:
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            continue
                    raise e
                else:
                    if attempt < max_tries - 1:
                        time.sleep(1)

        # fallback: zero vectors
        return [[0.0] * 768 for _ in texts]

    async def embed_single(self, text: str, model: str = None, max_tries: int = 5) -> List[float]:
        embeddings = self.embed([text], model, max_tries)
        return embeddings[0] if embeddings else [0.0] * 768
