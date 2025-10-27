import json
import logging
import os
import time
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

from backend.llm.base import BaseLLMClient
from backend.llm.factory import llm_registry
from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.graph_elements import Node, Edge

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiGraphExtractor(GraphExtractorBase):
    """Graph extractor for Gemini with multiple API key rotation support."""

    def __init__(self, api_key: str = None, model_name: str = None):
        if model_name is None:
            model_name = os.getenv("LLM_GRAPH_MODEL", "gemini-2.5-flash")

        # Load multiple API keys
        self.api_keys = self._load_api_keys(api_key)
        if not self.api_keys:
            raise ValueError(
                "No GEMINI_API_KEY found in environment variables. "
                "Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.")

        self.current_key_index = 0
        self.model_name = model_name
        self.llm_client = None
        self._initialize_client()

    def _load_api_keys(self, primary_key: str = None) -> List[str]:
        """Load multiple API keys from environment variables."""
        keys = []

        # Add primary key if provided
        if primary_key:
            keys.append(primary_key)

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

    def _initialize_client(self) -> None:
        """Initialize LLM client with current API key."""
        if not self.api_keys:
            raise ValueError("No API keys available")

        current_key = self.api_keys[self.current_key_index]
        logger.info(
            f"Initializing Gemini client with key {self.current_key_index + 1}/{len(self.api_keys)}")

        self.llm_client: BaseLLMClient = llm_registry.create_llm_client(
            provider_name="gemini",
            api_key=current_key,
            model_name=self.model_name,
        )

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

    def _generate_with_retry(self, max_tries: int, **kwargs) -> str:
        """Calls the LLM client's generate method with retry logic and API key rotation."""
        last_exception = None
        keys_tried = set()

        for attempt in range(max_tries):
            try:
                response = self.llm_client.generate(**kwargs)
                return response.message
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

    def extract_nodes(
            self,
            document_content: str,
            system_prompt: str = None,
            user_prompt_template: str = None,
            mode: str = "standard",
            k: int = 1,
            feedback_prompt_template: str = None,
            **kwargs) -> List[Node]:

        if not system_prompt or not user_prompt_template:
            logger.error(
                "System or user prompt not provided for node extraction.")
            return []

        node_schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["text", "description"]
                    }
                }
            },
            "required": ["entities"]
        }

        # The initial user prompt for both modes
        initial_user_prompt = user_prompt_template.format(
            text=document_content)
        max_tries = 5
        response_message = ""

        if mode == "interactive":
            if not feedback_prompt_template:
                logger.error(
                    "Feedback prompt template is required for interactive mode.")
                return []

            feedback_from_previous_run = "None"
            extraction_json_output = ""

            for i in range(k):

                # --- 1. Extraction Run (with schema) ---
                current_user_prompt = initial_user_prompt

                if i > 0:
                    current_user_prompt += f"\n\nUse the following feedback to improve your answer:\n{feedback_from_previous_run}"

                extraction_json_output = self._generate_with_retry(
                    max_tries=max_tries,
                    user_prompt=current_user_prompt,
                    system_prompt=system_prompt,
                    response_schema=node_schema  # Always use schema for extraction
                )

                if i == k:
                    break

                # --- 2. Feedback Run (no schema) ---
                feedback_user_prompt = feedback_prompt_template.format(
                    text=document_content,
                    previous_output=extraction_json_output
                )
                feedback_from_previous_run = self._generate_with_retry(
                    max_tries=max_tries,
                    user_prompt=feedback_user_prompt,
                    system_prompt=system_prompt,  # Re-use system prompt for context
                    response_schema=None  # No schema for feedback
                )

            response_message = extraction_json_output
        else:
            response_message = self._generate_with_retry(
                max_tries=max_tries,
                user_prompt=initial_user_prompt,
                system_prompt=system_prompt,
                response_schema=node_schema
            )

        # --- Final Parsing ---
        try:
            data = json.loads(response_message)
            nodes = []
            for entity in data.get("entities", []):
                node_text = entity.get("text")
                node_desc = entity.get("description")

                if not node_text or not node_desc:
                    raise ValueError(
                        f"Entity with missing 'text' or 'description' found: {entity}")

                nodes.append(Node(
                    id=node_text,
                    properties=entity
                ))
            return nodes
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(
                f"Failed to parse final JSON output or invalid entity found: {e}")
            logger.error(f"Problematic response: {response_message}")
            # Return empty list instead of raising error to continue processing
            logger.warning(
                "Returning empty nodes list due to JSON parsing error")
            return []

    def extract_edges(
            self,
            document_content: str,
            nodes: List[Node],
            system_prompt: str = None,
            user_prompt_template: str = None,
            **kwargs) -> List[Edge]:
        if not nodes:
            logger.info("No nodes provided, skipping edge extraction.")
            return []
        if not system_prompt or not user_prompt_template:
            logger.error(
                "System or user prompt not provided for edge extraction.")
            return []

        # Serialize nodes for the prompt
        nodes_str = "\n".join(
            [f"- {node.properties.get('text')} ({node.properties.get('description')})" for node in nodes])

        user_prompt = user_prompt_template.format(
            text=document_content, nodes=nodes_str)

        edge_schema = {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "label": {"type": "string"}
                        },
                        "required": ["source", "target", "label"]
                    }
                }
            },
            "required": ["relationships"]
        }

        response_message = self._generate_with_retry(
            max_tries=5,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=edge_schema
        )

        try:
            data = json.loads(response_message)
            edges = [Edge(**rel) for rel in data.get("relationships", [])]
            return edges
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse final JSON output for edges: {e}")
            logger.error(f"Problematic response: {response_message}")
            # Return empty list instead of raising error to continue processing
            logger.warning(
                "Returning empty edges list due to JSON parsing error")
            return []
