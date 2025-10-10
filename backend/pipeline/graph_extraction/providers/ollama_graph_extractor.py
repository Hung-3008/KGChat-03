import json
import logging
import os

from dotenv import load_dotenv

from backend.llm.base import BaseLLMClient
from backend.llm.factory import llm_registry
from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.graph_elements import Node, Edge
from typing import List, Dict

load_dotenv()
logger = logging.getLogger(__name__)


class OllamaGraphExtractor(GraphExtractorBase):
    """Graph extractor for Ollama."""

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

    def extract_nodes(
        self, 
        document_content: str, 
        system_prompt: str = None, 
        user_prompt_template: str = None,
        mode: str = "standard",
        k: int = 1,
        feedback_prompt_template: str = None,
        **kwargs) -> List[Node]:
        logger.info("--- Calling Ollama LLM for node extraction ---")

        if not system_prompt or not user_prompt_template:
            logger.error("System or user prompt not found for node extraction.")
            return []

        response_message = ""
        if mode == "interactive":
            if not feedback_prompt_template:
                logger.error("Feedback prompt template is required for interactive mode.")
                return []
            
            feedback_from_previous_run = "None"
            extraction_json_output = ""

            for i in range(k):
                
                # --- 1. Extraction Run (with JSON format) ---
                current_user_prompt = user_prompt_template.format(text=document_content)
                if i > 0:
                    current_user_prompt += f"\n\nUse the following feedback to improve your answer:\n{feedback_from_previous_run}"

                full_prompt = f"{system_prompt}\n\n{current_user_prompt}"
                response = self.llm_client.generate(user_prompt=full_prompt, format="json") # Always get JSON
                extraction_json_output = response.message

                print("\n\n Full Prompt:", full_prompt)

                if i == k:
                    break

                # --- 2. Feedback Run (no JSON format) ---
                feedback_user_prompt = feedback_prompt_template.format(
                    text=document_content, 
                    previous_output=extraction_json_output
                )
                full_feedback_prompt = f"{system_prompt}\n\n{feedback_user_prompt}"
                print("\n\nFull Feedback Prompt:", full_feedback_prompt)
                feedback_response = self.llm_client.generate(user_prompt=full_feedback_prompt, format="") # No format for feedback
                feedback_from_previous_run = feedback_response.message

            response_message = extraction_json_output
        else: # Standard mode
            user_prompt = user_prompt_template.format(text=document_content)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            print("\n\nFull Prompt:", full_prompt)  
            response = self.llm_client.generate(user_prompt=full_prompt, format="json")
            #response = self.llm_client.generate(user_prompt=full_prompt)

            print("\n\nLLM Raw Response:", response) 
            
            response_message = response.message

        try:
            logger.info(f"--- Final Ollama LLM Response --- \n{response_message}")
            
            data = json.loads(response_message)
            
            nodes = []
            for entity in data.get("entities", []):
                node_type = entity.get("type")
                if isinstance(node_type, list):
                    node_type = node_type[0] if node_type else None

                nodes.append(Node(
                    id=entity.get("text"),  # Or generate a unique ID
                    type=node_type,
                    properties=entity
                ))
            
            #logger.info(f"--- Ollama extracted {len(nodes)} nodes successfully ---")
            return nodes

        except (json.JSONDecodeError, AttributeError, Exception) as e:
            logger.error(f"An error occurred parsing the LLM response: {e}")
            logger.error(f"Problematic response: {response_message}")
            raise e

    def extract_edges(
        self, 
        document_content: str, 
        nodes: List[Node],
        system_prompt: str = None,
        user_prompt_template: str = None,
        **kwargs) -> List[Edge]:
        logger.info("--- Calling Ollama LLM for edge extraction ---")
        if not nodes:
            logger.info("No nodes provided, skipping edge extraction.")
            return []
        if not system_prompt or not user_prompt_template:
            logger.error("System or user prompt not provided for edge extraction.")
            return []

        # Serialize nodes for the prompt
        nodes_str = "\n".join([f"- {node.properties.get('text')} ({node.properties.get('type')})" for node in nodes])

        user_prompt = user_prompt_template.format(text=document_content, nodes=nodes_str)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self.llm_client.generate(user_prompt=full_prompt, format="json")
            response_message = response.message
        except Exception as e:
            logger.error(f"An error occurred during edge generation: {e}")
            raise e

        try:
            data = json.loads(response_message)
            edges = [Edge(**rel) for rel in data.get("relationships", [])]
            #logger.info(f"--- Ollama extracted {len(edges)} edges successfully ---")
            return edges
        except (json.JSONDecodeError, AttributeError, Exception) as e:
            logger.error(f"An error occurred parsing the LLM edge response: {e}")
            logger.error(f"Problematic response: {response_message}")
            raise e