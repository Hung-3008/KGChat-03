import google.generativeai as genai
from typing import Optional, Dict, Any, List
import logging
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .gemini_config import GeminiConfig

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.config: GeminiConfig = config
        self._model = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.config.api_key:
            raise AuthenticationError("Gemini API key is required")

        genai.configure(api_key=self.config.api_key)
        generation_config = self.config.get_generation_config()

        self._model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=generation_config
        )

        logger.info(
            f"Initialized Gemini client with model: {self.config.model_name}")

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        try:
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            else:
                full_prompt = user_prompt

            response = self._model.generate_content(full_prompt)

            message = response.text if response.parts else "No response generated"

            metadata = {
                "model": self.config.model_name,
                "finish_reason": getattr(response, 'finish_reason', None),
                "usage": getattr(response, 'usage_metadata', None)
            }

            return LLMResponse(message=message, metadata=metadata)

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Gemini API.

        Args:
            texts: List of input strings to embed.
            model: Model to use for embeddings (optional, uses config default if not provided)

        Returns:
            A list of embedding vectors, one for each input text.
        """
        if not texts:
            return []

        embedding_model = model or self.config.embedding_model

        try:
            embeddings = []

            # Gemini embedding API requires individual calls for each text
            for text in texts:
                # Use the embed_content method from genai
                result = genai.embed_content(
                    model=embedding_model,
                    content=text,
                    task_type="retrieval_document",  # Can be retrieval_document or retrieval_query
                    title=None  # Optional title for the content
                )

                # Extract the embedding values
                embedding = result['embedding']
                embeddings.append(embedding)

            logger.info(
                f"Generated {len(embeddings)} embeddings using {embedding_model}")
            return embeddings

        except Exception as e:
            logger.warning(
                f"Gemini embedding failed: {e}. Falling back to zero vectors.")

            # Fallback to zero vectors if API fails
            embeddings = []
            for _ in texts:
                # Create a zero vector of standard embedding dimension (768 for text-embedding-004)
                embedding = [0.0] * 768
                embeddings.append(embedding)

            return embeddings

    async def embed_single(self, text: str, model: str = None) -> List[float]:
        """
        Generate embedding for a single text using Gemini API.

        Args:
            text: The input string to embed.
            model: Model to use for embeddings (optional)

        Returns:
            A list of float numbers representing the embedding vector.
        """
        embeddings = self.embed([text], model)
        return embeddings[0] if embeddings else [0.0] * 768
