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
    """Graph extractor for Gemini."""

    def __init__(self, api_key: str = None, model_name: str = None, prompt_settings: Dict = None):
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
        )
        
        self.prompts = {}
        if prompt_settings:
            base_path = prompt_settings.get("base_path", "prompts")
            stages = prompt_settings.get("stages", {})
            for stage, stage_prompts in stages.items():
                self.prompts[stage] = {}
                for prompt_name, prompt_value in stage_prompts.items():
                    if isinstance(prompt_value, str) and prompt_value.endswith('.txt'):
                        with open(os.path.join(base_path, prompt_value), "r") as f:
                            self.prompts[stage][prompt_name] = f.read()
                    else:
                        self.prompts[stage][prompt_name] = prompt_value

    def extract_nodes(self, document_content: str, system_prompt: str = None, user_prompt_template: str = None) -> List[Node]:
        logger.info("--- Calling Gemini LLM for node extraction ---")
        
        stage = "node_extraction"
        if stage not in self.prompts and (not system_prompt or not user_prompt_template):
            logger.error(f"Prompts for stage '{stage}' not found and no override provided.")
            return []

        stage_prompts = self.prompts.get(stage, {})
        system_prompt = system_prompt or stage_prompts.get("system_prompt")
        user_prompt_template = user_prompt_template or stage_prompts.get("user_prompt_template")

        if not system_prompt or not user_prompt_template:
            logger.error(f"System or user prompt not found for stage '{stage}'.")
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
                            "type": {"type": "string", "enum": ["SpecificDisease", "DiseaseClass", "Modifier"]},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"}
                        },
                        "required": ["text", "type", "start", "end"]
                    }
                }
            },
            "required": ["entities"]
        }

        user_prompt = user_prompt_template.format(text=document_content)
        
        max_tries = 3
        response_message = ""
        for i in range(max_tries):
            try:
                response = self.llm_client.generate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    response_schema=node_schema
                )
                response_message = response.message
                logger.info(f"--- Gemini LLM Response --- \n{response_message}")
                
                data = json.loads(response_message)
                
                nodes = []
                for entity in data.get("entities", []):
                    nodes.append(Node(
                        id=entity.get("text"), # Or generate a unique ID
                        type=entity.get("type"),
                        properties=entity
                    ))
                
                logger.info("--- Gemini JSON parsing and node creation successful ---")
                return nodes

            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Attempt {i+1}/{max_tries} failed with Gemini client: {e}")
                if i < max_tries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Problematic response: {response_message}")
            except Exception as e:
                logger.error(f"Attempt {i+1}/{max_tries} failed with an unexpected error: {e}")
                if i < max_tries - 1:
                    time.sleep(1)
        
        return []

    def extract_edges(self, document_content: str) -> List[Edge]:
        logger.info("--- Calling Gemini LLM for edge extraction ---")
        
        stage = "edge_extraction"
        if stage not in self.prompts:
            logger.error(f"Prompts for stage '{stage}' not found.")
            return []

        system_prompt = self.prompts[stage].get("system_prompt")
        user_prompt_template = self.prompts[stage].get("user_prompt_template")

        if not system_prompt or not user_prompt_template:
            logger.error(f"System or user prompt not found for stage '{stage}'.")
            return []

        edge_schema = {
            "type": "object",
            "properties": {
                "edges": {
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
            "required": ["edges"]
        }

        user_prompt = user_prompt_template.format(text=document_content)

        max_tries = 3
        for i in range(max_tries):
            try:
                response = self.llm_client.generate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    response_schema=edge_schema
                )
                response_message = response.message
                logger.info(f"--- Gemini LLM Response --- \n{response_message}")
                
                data = json.loads(response_message)
                
                edges = []
                for edge in data.get("edges", []):
                    edges.append(Edge(
                        source=edge.get("source"),
                        target=edge.get("target"),
                        label=edge.get("label"),
                        properties=edge
                    ))
                
                logger.info("--- Gemini JSON parsing and edge creation successful ---")
                return edges

            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Attempt {i+1}/{max_tries} failed with Gemini client: {e}")
                if i < max_tries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Problematic response: {response_message}")
            except Exception as e:
                logger.error(f"Attempt {i+1}/{max_tries} failed with an unexpected error: {e}")
                if i < max_tries - 1:
                    time.sleep(1)

        return []
