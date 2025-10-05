import json
import logging
import os
from typing import List

from dotenv import load_dotenv

from backend.llm.base import BaseLLMClient
from backend.llm.factory import llm_registry
from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.graph_elements import Node

load_dotenv()
logger = logging.getLogger(__name__)


class OllamaGraphExtractor(GraphExtractorBase):
    """A sample graph extractor for Ollama to test LLM connectivity."""

    def __init__(self, model_name: str = None, api_url: str = None):
        if model_name is None:
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        if api_url is None:
            api_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm_client: BaseLLMClient = llm_registry.create_llm_client(
            provider_name="ollama",
            model_name=model_name,
            api_url=api_url,
        )
        self.prompt_template = (
            "Return a JSON with a single key 'status' and value 'ok'."
        )

    def extract_nodes(self, document_content: str) -> List[Node]:
        """Calls the LLM and returns a dummy response."""
        logger.info("--- Calling Ollama LLM ---")
        try:
            response = self.llm_client.generate(user_prompt=self.prompt_template)
            logger.info(f"--- Ollama LLM Response --- \n{response.message}")
            json.loads(response.message)
            logger.info("--- Ollama JSON parsing successful ---")
        except Exception as e:
            logger.error(f"An error occurred with Ollama client: {e}")
        return []