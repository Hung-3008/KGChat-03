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


class GeminiGraphExtractor(GraphExtractorBase):
    """A sample graph extractor for Gemini to test LLM connectivity."""

    def __init__(self, api_key: str = None, model_name: str = None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        if model_name is None:
            model_name = os.getenv("LLM_GRAPH_MODEL", "gemini-2.5-flash")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.llm_client: BaseLLMClient = llm_registry.create_llm_client(
            provider_name="gemini",
            api_key=api_key,
            model_name=model_name,
            response_mime_type="application/json",
        )
        self.prompt_template = (
            "Return a JSON with a single key 'status' and value 'ok'."
        )

    def extract_nodes(self, document_content: str) -> List[Node]:
        logger.info("--- Calling Gemini LLM ---")
        try:
            response = self.llm_client.generate(user_prompt=self.prompt_template)
            logger.info(f"--- Gemini LLM Response --- \n{response.message}")
            json.loads(response.message)
            logger.info("--- Gemini JSON parsing successful ---")
        except Exception as e:
            logger.error(f"An error occurred with Gemini client: {e}")
        return []
    


    def extract_edges(self, document_content: str):
        pass


    def embddings(self, text: str):
        pass
