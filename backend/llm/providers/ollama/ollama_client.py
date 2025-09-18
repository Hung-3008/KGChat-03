import ollama
from typing import Optional, Dict, Any, List
import logging
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from ollama import AsyncClient, ChatResponse
from .ollama_config import OllamaConfig



logger = logging.getLogger(__name__)


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
        logger.info(f"Initialized Ollama client with host: {self.config.host}")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = self._client.chat(
                model=self.config.model_name,
                messages=messages,
                **self._generation_config
            )

            if not response or "message" not in response:
                raise RuntimeError("Ollama response invalid or missing 'message'")

            message = response['message']['content'] if response.get("message") else "No response generated"

            metadata = {
                "model": self.config.model_name,
                "finish_reason": response.get("message", {}).get("stop_reason"),
                "usage": {
                    "input_tokens": response.get("prompt_eval_count"),
                    "output_tokens": response.get("eval_count")
                }
            }

            return LLMResponse(message=message, metadata=metadata)

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Ollama API.

        Args:
            text: The input string to embed.

        Returns:
            A list of float numbers representing the embedding vector.
        """
        try:
            model = self.config.model_name
            logger.debug(f"Generating embedding with model: {model}")

            response = self._client.embed(
                model=model,
                input=text
            )

            embedding = response.get("embedding")
            if embedding is None:
                raise ValueError("Embedding not found in response")

            return embedding

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error during embedding: {e.status_code} - {e.error}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {str(e)}")
            raise
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the Ollama server.

        Returns:
            List of models with metadata.
        """
        try:
            logger.debug("Fetching available models from Ollama")
            response = self._client.list()
            return response.get("models", [])
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error when listing models: {e.status_code} - {e.error}")
            return []
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            return []     

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama registry.

        Args:
            model_name: The name of the model to pull.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info(f"Pulling model '{model_name}' from Ollama")
            self._client.pull(model=model_name)
            return True
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error when pulling model: {e.status_code} - {e.error}")
            return False
        except Exception as e:
            logger.error(f"Error pulling model '{model_name}': {str(e)}")
            return False
    
    def create_model(
        self,
        model_name: str,
        base_model: str,
        system_prompt: str,
        **kwargs
    ) -> bool:
        """
        Create a new model from a base model with a custom system prompt.

        Args:
            model_name: Name for the new model.
            base_model: Existing model to build from.
            system_prompt: Custom behavior prompt.
            **kwargs: Additional parameters.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info(f"Creating new model '{model_name}' from base model '{base_model}'")
            self._client.create(
                model=model_name,
                from_=base_model,
                system=system_prompt,
                **kwargs
            )
            return True
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error during model creation: {e.status_code} - {e.error}")
            return False
        except Exception as e:
            logger.error(f"Error creating model '{model_name}': {str(e)}")
            return False

            
            