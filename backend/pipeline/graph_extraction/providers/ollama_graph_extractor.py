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


class OllamaGraphExtractor(GraphExtractorBase):
    """Graph extractor for Ollama."""

    def __init__(self, model_name: str = None, api_url: str = None, prompt_settings: Dict = None):
        if model_name is None:
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        if api_url is None:
            api_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm_client: BaseLLMClient = llm_registry.create_llm_client(
            provider_name="ollama",
            model_name=model_name,
            api_url=api_url,
        )
        
        self.prompts = {}
        if prompt_settings:
            base_path = prompt_settings.get("base_path", "prompts")
            stages = prompt_settings.get("stages", {})
            for stage, stage_prompts in stages.items():
                self.prompts[stage] = {}
                for prompt_name, value in stage_prompts.items():
                    if isinstance(value, str) and ".txt" in value:
                        try:
                            with open(os.path.join(base_path, value), "r") as f:
                                self.prompts[stage][prompt_name] = f.read()
                        except FileNotFoundError:
                            logger.warning(f"Prompt file not found for {prompt_name} in stage {stage} at path: {os.path.join(base_path, value)}")
                            self.prompts[stage][prompt_name] = None # Or handle as an error
                    else:
                        self.prompts[stage][prompt_name] = value

    def extract_nodes(self, document_content: str, system_prompt: str = None, user_prompt_template: str = None) -> List[Node]:
        logger.info("--- Calling Ollama LLM for node extraction ---")
        
        stage = "node_extraction"
        if stage not in self.prompts and (not system_prompt or not user_prompt_template):
            logger.error(f"Prompts for stage '{stage}' not found and no override provided.")
            return []

        stage_prompts = self.prompts.get(stage, {})
        mode = stage_prompts.get("mode", "standard")
        k = stage_prompts.get("k", 1)

        system_prompt = system_prompt or stage_prompts.get("system_prompt")
        user_prompt_template = user_prompt_template or stage_prompts.get("user_prompt_template")

        if not system_prompt or not user_prompt_template:
            logger.error(f"System or user prompt not found for stage '{stage}'.")
            return []

        user_prompt = user_prompt_template.format(text=document_content)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response_message = ""

        if mode == "interactive":
            feedback_prompt_template = stage_prompts.get("feedback_prompt")
            if not feedback_prompt_template:
                logger.error("Feedback prompt not found for interactive mode.")
                return []

            for i in tqdm(range(k), desc="--- Interactive Mode ---"):
                try:
                    response = self.llm_client.generate(user_prompt=full_prompt, format="json")
                    response_message = response.message
                    
                    # For feedback, we don't need JSON format.
                    feedback_prompt = feedback_prompt_template.format(text=document_content, previous_output=response_message)
                    feedback_response = self.llm_client.generate(user_prompt=feedback_prompt)
                    feedback = feedback_response.message

                    full_prompt += f"\n\nPrevious output:\n{response_message}\n\nFeedback:\n{feedback}"

                except Exception as e:
                    logger.error(f"An error occurred during interactive generation: {e}")
                    return []
        else: # standard mode
            try:
                response = self.llm_client.generate(user_prompt=full_prompt, format="json")
                response_message = response.message
            except Exception as e:
                logger.error(f"An error occurred during standard generation: {e}")
                return []

        try:
            logger.info(f"--- Ollama LLM Response --- \n{response_message}")
            
            data = json.loads(response_message)
            
            nodes = []
            for entity in data.get("entities", []):
                node_type = entity.get("type")
                if isinstance(node_type, list):
                    node_type = node_type[0] if node_type else None

                nodes.append(Node(
                    id=entity.get("text"), # Or generate a unique ID
                    type=node_type,
                    properties=entity
                ))
            
            logger.info("--- Ollama JSON parsing and node creation successful ---")
            return nodes

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"An error occurred with Ollama client: {e}")
            logger.error(f"Problematic response: {response_message}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []

    def extract_edges(self, document_content: str) -> List[Edge]:
        logger.info("--- Calling Ollama LLM for edge extraction ---")
        
        stage = "edge_extraction"
        if stage not in self.prompts:
            logger.error(f"Prompts for stage '{stage}' not found.")
            return []

        system_prompt = self.prompts[stage].get("system_prompt")
        user_prompt_template = self.prompts[stage].get("user_prompt_template")

        if not system_prompt or not user_prompt_template:
            logger.error(f"System or user prompt not found for stage '{stage}'.")
            return []

        user_prompt = user_prompt_template.format(text=document_content)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self.llm_client.generate(user_prompt=full_prompt, format="json")
            logger.info(f"--- Ollama LLM Response --- \n{response.message}")
            
            data = json.loads(response.message)
            
            edges = []
            for edge in data.get("edges", []):
                edges.append(Edge(
                    source=edge.get("source"),
                    target=edge.get("target"),
                    label=edge.get("label"),
                    properties=edge
                ))
            
            logger.info("--- Ollama JSON parsing and edge creation successful ---")
            return edges

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"An error occurred with Ollama client: {e}")
            logger.error(f"Problematic response: {response.message}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []