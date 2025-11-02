import json
import os
from typing import Union

from backend.llm.base import BaseLLMClient
from backend.llm.factory import llm_registry
from backend.pipeline.graph_extraction.base import GraphExtractorBase


class OllamaGraphExtractor(GraphExtractorBase):

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

    def _call_llm(self, user_prompt: str, system_prompt: str = None, response_schema: dict = None, max_tries: int = 1, **kwargs) -> Union[str, dict]:
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        fmt = "json" if response_schema is not None else ""
        response = self.llm_client.generate(user_prompt=full_prompt, format=fmt)
        if response_schema and fmt == "json":
            try:
                return json.loads(response.message)
            except json.JSONDecodeError:
                return response.message
        return response.message

    def extract_nodes(self, document_content: str, system_prompt: str = None, user_prompt_template: str = None, mode: str = "standard", k: int = 1, feedback_prompt_template: str = None, **kwargs):
        node_schema = {}
        return super().extract_nodes(document_content, system_prompt=system_prompt, user_prompt_template=user_prompt_template, mode=mode, k=k, feedback_prompt_template=feedback_prompt_template, response_schema=node_schema, **kwargs)

    def extract_edges(self, document_content: str, nodes, system_prompt: str = None, user_prompt_template: str = None, **kwargs):
        edge_schema = {}
        return super().extract_edges(document_content, nodes, system_prompt=system_prompt, user_prompt_template=user_prompt_template, response_schema=edge_schema, **kwargs)