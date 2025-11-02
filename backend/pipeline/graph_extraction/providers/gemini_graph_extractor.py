import json
import os
import time
from typing import List, Dict, Union

from backend.llm.base import BaseLLMClient
from backend.llm.factory import llm_registry
from backend.pipeline.graph_extraction.base import GraphExtractorBase


class GeminiGraphExtractor(GraphExtractorBase):

    def __init__(self, api_key: str = None, model_name: str = None):
        if model_name is None:
            model_name = os.getenv("LLM_GRAPH_MODEL", "gemini-2.5-flash")

        self.api_keys = self._load_api_keys(api_key)
        if not self.api_keys:
            raise ValueError(
                "No GEMINI_API_KEY found in environment variables. Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.")

        self.current_key_index = 0
        self.model_name = model_name
        self.llm_client = None
        self._initialize_client()

    def _load_api_keys(self, primary_key: str = None) -> List[str]:
        keys = []
        if primary_key:
            keys.append(primary_key)
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                keys.append(key)
                i += 1
            else:
                break
        legacy_key = os.getenv("GOOGLE_API_KEY")
        if legacy_key and legacy_key not in keys:
            keys.append(legacy_key)
        return keys

    def _initialize_client(self) -> None:
        if not self.api_keys:
            raise ValueError("No API keys available")
        current_key = self.api_keys[self.current_key_index]
        self.llm_client: BaseLLMClient = llm_registry.create_llm_client(
            provider_name="gemini",
            api_key=current_key,
            model_name=self.model_name,
        )

    def _rotate_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        try:
            self._initialize_client()
            return True
        except Exception:
            return False

    def _generate_with_retry(self, max_tries: int, **kwargs) -> Union[str, dict]:
        last_exception = None
        keys_tried = set()
        for attempt in range(max_tries):
            try:
                response = self.llm_client.generate(**kwargs)
                # If response_schema is provided and the response is JSON, try to parse it
                if kwargs.get('response_schema') and isinstance(response.message, str):
                    try:
                        return json.loads(response.message)
                    except json.JSONDecodeError:
                        pass
                return response.message
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["quota", "rate", "limit", "exceeded", "billing"]):
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            continue
                        else:
                            pass
                    else:
                        if attempt < max_tries - 1:
                            wait_time = 2 ** attempt
                            time.sleep(wait_time)
                elif "authentication" in error_str or "api_key" in error_str:
                    if len(keys_tried) < len(self.api_keys):
                        if self._rotate_api_key():
                            keys_tried.add(self.current_key_index)
                            continue
                    raise e
                else:
                    if attempt < max_tries - 1:
                        time.sleep(1)
        raise last_exception

    def _call_llm(self, user_prompt: str, system_prompt: str = None, response_schema: dict = None, max_tries: int = 5, **kwargs) -> Union[str, dict]:
        prompt = user_prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_prompt}"
        return self._generate_with_retry(max_tries=max_tries, user_prompt=prompt, system_prompt=system_prompt, response_schema=response_schema)

    def extract_nodes(self, document_content: str, system_prompt: str = None, user_prompt_template: str = None, mode: str = "standard", k: int = 1, feedback_prompt_template: str = None, **kwargs) -> List[object]:
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
        return super().extract_nodes(document_content, system_prompt=system_prompt, user_prompt_template=user_prompt_template, mode=mode, k=k, feedback_prompt_template=feedback_prompt_template, response_schema=node_schema, **kwargs)

    def extract_edges(self, document_content: str, nodes: List[object], system_prompt: str = None, user_prompt_template: str = None, **kwargs) -> List[object]:
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
        return super().extract_edges(document_content, nodes, system_prompt=system_prompt, user_prompt_template=user_prompt_template, response_schema=edge_schema, **kwargs)
