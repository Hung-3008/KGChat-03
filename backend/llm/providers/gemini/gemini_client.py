import google.generativeai as genai
from typing import Optional, Dict, Any, List
import logging
import os
import time
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .gemini_config import GeminiConfig

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.config: GeminiConfig = config
        self._model = None

        # Load multiple API keys for rotation
        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            raise AuthenticationError(
                "No GEMINI_API_KEY found in environment variables")

        self.current_key_index = 0
        self._initialize_client()

    def _load_api_keys(self) -> List[str]:
        """Load multiple API keys from environment variables."""
        keys = []

        # Add primary key if provided
        if self.config.api_key:
            keys.append(self.config.api_key)

        # Load numbered keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...)
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                keys.append(key)
                i += 1
            else:
                break

        # Also check for legacy GOOGLE_API_KEY
        legacy_key = os.getenv("GOOGLE_API_KEY")
        if legacy_key and legacy_key not in keys:
            keys.append(legacy_key)

        return keys

    def _rotate_api_key(self) -> bool:
        """Rotate to next available API key."""
        if len(self.api_keys) <= 1:
            logger.warning("Only one API key available, cannot rotate")
            return False

        self.current_key_index = (
            self.current_key_index + 1) % len(self.api_keys)
        logger.info(
            f"Rotating to API key {self.current_key_index + 1}/{len(self.api_keys)}")

        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize with new API key: {e}")
            return False

    def _initialize_client(self) -> None:
        if not self.api_keys:
            raise AuthenticationError("No API keys available")

        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        logger.info(
            f"Initialized Gemini client with key {self.current_key_index + 1}/{len(self.api_keys)} and model: {self.config.model_name}")

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Dict[str, Any]] = None, max_tries: int = 5, **kwargs) -> LLMResponse:
        """Generate content with retry logic and API key rotation."""
        last_exception = None
        keys_tried = set()

        for attempt in range(max_tries):
            try:
                model = genai.GenerativeModel(
                    model_name=self.config.model_name,
                    system_instruction=system_prompt
                )

                generation_config_args = self.config.get_generation_config()

                if response_schema:
                    generation_config_args["response_mime_type"] = "application/json"
                    generation_config_args["response_schema"] = response_schema
                elif self.config.response_mime_type:
                    generation_config_args["response_mime_type"] = self.config.response_mime_type

                generation_config = genai.GenerationConfig(
                    **generation_config_args)

                response = model.generate_content(
                    user_prompt,
                    generation_config=generation_config
                )

                message = response.text if response.parts else "No response generated"

                metadata = {
                    "model": self.config.model_name,
                    "finish_reason": getattr(response, 'finish_reason', None),
                    "usage": getattr(response, 'usage_metadata', None)
                }

                return LLMResponse(message=message, metadata=metadata)

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if this is a quota/rate limit error
                if any(keyword in error_str for keyword in ["quota", "rate", "limit", "exceeded", "billing"]):
                    logger.warning(
                        f"Rate limit/quota error on attempt {attempt + 1}/{max_tries}: {e}")

                    # Try rotating API key if we haven't tried all keys yet
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            logger.info(
                                f"Switched to API key {self.current_key_index + 1}, retrying...")
                            continue
                        else:
                            logger.error("Failed to rotate to next API key")
                    else:
                        logger.warning(
                            "All API keys have been tried, using exponential backoff")
                        if attempt < max_tries - 1:
                            wait_time = 2 ** attempt
                            logger.info(
                                f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                elif "authentication" in error_str or "api_key" in error_str:
                    logger.error(f"Authentication error: {e}")
                    # Try rotating API key for auth errors too
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            logger.info(
                                f"Switched to API key {self.current_key_index + 1} due to auth error")
                            continue
                    raise e  # Don't retry if all keys failed auth
                else:
                    logger.warning(
                        f"LLM generation attempt {attempt + 1}/{max_tries} failed: {e}")
                    if attempt < max_tries - 1:
                        time.sleep(1)  # Wait 1 second before retrying

        logger.error(
            f"LLM generation failed after {max_tries} attempts with all available API keys.")
        raise last_exception

    def get_api_key_info(self) -> Dict[str, any]:
        """Get information about available API keys."""
        return {
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_key_index,
            "current_key_preview": self.api_keys[self.current_key_index][:10] + "..." if self.api_keys else None,
            "all_keys_preview": [key[:10] + "..." for key in self.api_keys]
        }

    def list_api_keys(self) -> None:
        """Print information about available API keys."""
        info = self.get_api_key_info()
        logger.info(f"API Key Status: {info['total_keys']} keys available")
        logger.info(
            f"Current key: {info['current_key_index'] + 1}/{info['total_keys']}")
        for i, key_preview in enumerate(info['all_keys_preview']):
            status = " (CURRENT)" if i == info['current_key_index'] else ""
            logger.info(f"  Key {i+1}: {key_preview}{status}")

    def embed(self, texts: List[str], model: str = None, max_tries: int = 5) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Gemini API with retry logic and API key rotation.

        Args:
            texts: List of input strings to embed.
            model: Model to use for embeddings (optional, uses config default if not provided)
            max_tries: Maximum number of retry attempts

        Returns:
            A list of embedding vectors, one for each input text.
        """
        if not texts:
            return []

        embedding_model = model or self.config.embedding_model
        last_exception = None
        keys_tried = set()

        for attempt in range(max_tries):
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
                last_exception = e
                error_str = str(e).lower()

                # Check if this is a quota/rate limit error
                if any(keyword in error_str for keyword in ["quota", "rate", "limit", "exceeded", "billing"]):
                    logger.warning(
                        f"Rate limit/quota error on embedding attempt {attempt + 1}/{max_tries}: {e}")

                    # Try rotating API key if we haven't tried all keys yet
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            logger.info(
                                f"Switched to API key {self.current_key_index + 1} for embeddings, retrying...")
                            continue
                        else:
                            logger.error(
                                "Failed to rotate to next API key for embeddings")
                    else:
                        logger.warning(
                            "All API keys have been tried for embeddings, using exponential backoff")
                        if attempt < max_tries - 1:
                            wait_time = 2 ** attempt
                            logger.info(
                                f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                elif "authentication" in error_str or "api_key" in error_str:
                    logger.error(f"Authentication error in embeddings: {e}")
                    # Try rotating API key for auth errors too
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            logger.info(
                                f"Switched to API key {self.current_key_index + 1} due to auth error in embeddings")
                            continue
                    raise e  # Don't retry if all keys failed auth
                else:
                    logger.warning(
                        f"Embedding attempt {attempt + 1}/{max_tries} failed: {e}")
                    if attempt < max_tries - 1:
                        time.sleep(1)  # Wait 1 second before retrying

        # If all retries failed, fall back to zero vectors
        logger.warning(
            f"All embedding attempts failed after {max_tries} tries with all available API keys. Falling back to zero vectors.")

        # Fallback to zero vectors if API fails
        embeddings = []
        for _ in texts:
            # Create a zero vector of standard embedding dimension (768 for text-embedding-004)
            embedding = [0.0] * 768
            embeddings.append(embedding)

        return embeddings

    async def embed_single(self, text: str, model: str = None, max_tries: int = 5) -> List[float]:
        """
        Generate embedding for a single text using Gemini API with retry logic.

        Args:
            text: The input string to embed.
            model: Model to use for embeddings (optional)
            max_tries: Maximum number of retry attempts

        Returns:
            A list of float numbers representing the embedding vector.
        """
        embeddings = self.embed([text], model, max_tries)
        return embeddings[0] if embeddings else [0.0] * 768
