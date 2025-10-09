import json
import logging
import os
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
    """Graph extractor for Gemini."""

    def __init__(self, api_key: str = None, model_name: str = None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        if model_name is None:
            model_name = os.getenv("LLM_GRAPH_MODEL", "gemini-1.5-flash")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.llm_client: BaseLLMClient = llm_registry.create_llm_client(
            provider_name="gemini",
            api_key=api_key,
            model_name=model_name,
        )

    def _generate_with_retry(self, max_tries: int, **kwargs) -> str:
        """Calls the LLM client's generate method with retry logic."""
        last_exception = None
        for attempt in range(max_tries):
            try:
                response = self.llm_client.generate(**kwargs)
                return response.message
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"LLM generation attempt {attempt + 1}/{max_tries} failed: {e}"
                )
                if attempt < max_tries - 1:
                    time.sleep(1)  # Wait 1 second before retrying

        logger.error(
            f"LLM generation failed after {max_tries} attempts."
        )
        raise last_exception


    def extract_nodes(
        self, 
        document_content: str, 
        system_prompt: str = None, 
        user_prompt_template: str = None,
        mode: str = "standard",
        k: int = 1,
        feedback_prompt_template: str = None,
        **kwargs) -> List[Node]:
        #logger.info("--- Calling Gemini LLM for node extraction ---")
        
        if not system_prompt or not user_prompt_template:
            logger.error("System or user prompt not provided for node extraction.")
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
                            "type": {"type": "string", "enum": ["SpecificDisease", "DiseaseClass", "Modifier"]}
                        },
                        "required": ["text", "type"]
                    }
                }
            },
            "required": ["entities"]
        }

        # The initial user prompt for both modes
        initial_user_prompt = user_prompt_template.format(text=document_content)
        max_tries = 3
        response_message = ""

        if mode == "interactive":
            if not feedback_prompt_template:
                logger.error("Feedback prompt template is required for interactive mode.")
                return []
            
            #logger.info(f"--- Running in Interactive Mode for {k} iterations ---")
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
                    response_schema=node_schema # Always use schema for extraction
                )

                if i == k - 1:
                    break

                # --- 2. Feedback Run (no schema) ---
                feedback_user_prompt = feedback_prompt_template.format(
                    text=document_content, 
                    previous_output=extraction_json_output
                )
                feedback_from_previous_run = self._generate_with_retry(
                    max_tries=max_tries,
                    user_prompt=feedback_user_prompt,
                    system_prompt=system_prompt, # Re-use system prompt for context
                    response_schema=None # No schema for feedback
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
                nodes.append(Node(
                    id=entity.get("text"),
                    type=entity.get("type"),
                    properties=entity
                ))
            #logger.info(f"--- Gemini extracted {len(nodes)} nodes successfully ---")
            return nodes
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse final JSON output: {e}")
            logger.error(f"Problematic response: {response_message}")
            raise e

    def extract_edges(
        self, 
        document_content: str, 
        nodes: List[Node],
        system_prompt: str = None,
        user_prompt_template: str = None,
        **kwargs) -> List[Edge]:
        #logger.info("--- Calling Gemini LLM for edge extraction ---")
        if not nodes:
            logger.info("No nodes provided, skipping edge extraction.")
            return []
        if not system_prompt or not user_prompt_template:
            logger.error("System or user prompt not provided for edge extraction.")
            return []

        # Serialize nodes for the prompt
        nodes_str = "\n".join([f"- {node.properties.get('text')} ({node.properties.get('type')})" for node in nodes])

        user_prompt = user_prompt_template.format(text=document_content, nodes=nodes_str)

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
            max_tries=3,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=edge_schema
        )

        try:
            data = json.loads(response_message)
            edges = [Edge(**rel) for rel in data.get("relationships", [])]
            #logger.info(f"--- Gemini extracted {len(edges)} edges successfully ---")
            return edges
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse final JSON output for edges: {e}")
            logger.error(f"Problematic response: {response_message}")
            raise e
