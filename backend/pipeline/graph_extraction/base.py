from abc import ABC, abstractmethod
from typing import List, Optional, Union
import json
from .graph_elements import Node, Edge, Graph
from backend.llm.factory import llm_registry


class GraphExtractorBase(ABC):

    def _generate_node_embeddings(self, nodes: List[Node], embedding_provider: str = "ollama", embedding_model: str = None) -> List[Node]:
        if not nodes:
            return nodes
        
        texts_to_embed = []
        for node in nodes:
            text = node.properties.get('text', '')
            texts_to_embed.append(text)
        
        try:
            embeddings = llm_registry.embed_texts(
                texts=texts_to_embed,
                provider_name=embedding_provider,
                model=embedding_model
            )
            
            for node, embedding in zip(nodes, embeddings):
                node.properties['embedding'] = embedding
                
        except Exception:
            pass
        
        return nodes

    def _parse_nodes_from_json(self, response_message: Union[str, dict]) -> List[Node]:
        if isinstance(response_message, str):
            response_str = response_message.strip()
            if response_str.startswith('```json'):
                response_str = response_str[7:]
            if response_str.startswith('```'):
                response_str = response_str[3:]
            if response_str.endswith('```'):
                response_str = response_str[:-3]
            response_str = response_str.strip()
            
            data = json.loads(response_str)
        else:
            data = response_message
        nodes = []
        for entity in data.get("entities", []):
            node_text = entity.get("text")
            node_desc = entity.get("description")
            if not node_text or not node_desc:
                raise ValueError(f"Entity with missing 'text' or 'description' found: {entity}")
            nodes.append(Node(id=node_text, properties=entity))
        return nodes

    def _parse_edges_from_json(self, response_message: Union[str, dict]) -> List[Edge]:
        if isinstance(response_message, str):
            response_str = response_message.strip()
            if response_str.startswith('```json'):
                response_str = response_str[7:]  
            if response_str.startswith('```'):
                response_str = response_str[3:]  
            if response_str.endswith('```'):
                response_str = response_str[:-3]  
            response_str = response_str.strip()
            
            data = json.loads(response_str)
        else:
            data = response_message
        return [Edge(**rel) for rel in data.get("relationships", [])]

    def _run_interactive_extraction(self, initial_user_prompt: str, system_prompt: Optional[str], feedback_prompt_template: Optional[str], k: int, response_schema: Optional[dict], max_tries: int = 5) -> Union[str, dict]:
        feedback_from_previous_run = "None"
        extraction_json_output = ""
        for i in range(k):
            current_user_prompt = initial_user_prompt
            if i > 0:
                current_user_prompt += f"\n\nUse the following feedback to improve your answer:\n{feedback_from_previous_run}"

            extraction_json_output = self._call_llm(user_prompt=current_user_prompt, system_prompt=system_prompt, response_schema=response_schema, max_tries=max_tries)

            if i == k:
                break

            if not feedback_prompt_template:
                break

            previous_output_str = json.dumps(extraction_json_output) if isinstance(extraction_json_output, dict) else extraction_json_output
            if isinstance(previous_output_str, str):
                previous_output_str = previous_output_str.strip()
                if previous_output_str.startswith('```json'):
                    previous_output_str = previous_output_str[7:]
                if previous_output_str.startswith('```'):
                    previous_output_str = previous_output_str[3:]
                if previous_output_str.endswith('```'):
                    previous_output_str = previous_output_str[:-3]
                previous_output_str = previous_output_str.strip()
            feedback_user_prompt = feedback_prompt_template.format(text=initial_user_prompt, previous_output=previous_output_str)
            feedback_response = self._call_llm(user_prompt=feedback_user_prompt, system_prompt=system_prompt, response_schema=None, max_tries=max_tries)
            feedback_from_previous_run = json.dumps(feedback_response) if isinstance(feedback_response, dict) else feedback_response
            if isinstance(feedback_from_previous_run, str):
                feedback_from_previous_run = feedback_from_previous_run.strip()
                if feedback_from_previous_run.startswith('```json'):
                    feedback_from_previous_run = feedback_from_previous_run[7:]
                if feedback_from_previous_run.startswith('```'):
                    feedback_from_previous_run = feedback_from_previous_run[3:]
                if feedback_from_previous_run.endswith('```'):
                    feedback_from_previous_run = feedback_from_previous_run[:-3]
                feedback_from_previous_run = feedback_from_previous_run.strip()

        return extraction_json_output

    def extract_nodes(self, document_content: str, system_prompt: Optional[str] = None, user_prompt_template: Optional[str] = None, mode: str = "standard", k: int = 1, feedback_prompt_template: Optional[str] = None, response_schema: Optional[dict] = None, max_tries: int = 5, **kwargs) -> List[Node]:
        if not system_prompt or not user_prompt_template:
            return []

        initial_user_prompt = user_prompt_template.format(text=document_content)

        if mode == "interactive":
            if not feedback_prompt_template:
                return []
            response_message = self._run_interactive_extraction(initial_user_prompt, system_prompt, feedback_prompt_template, k, response_schema, max_tries=max_tries)
            
        else:
            response_message = self._call_llm(user_prompt=initial_user_prompt, system_prompt=system_prompt, response_schema=response_schema, max_tries=max_tries)

        try:
            nodes = self._parse_nodes_from_json(response_message)
            embedding_provider = kwargs.get('embedding_provider', 'ollama')
            embedding_model = kwargs.get('embedding_model')
            nodes = self._generate_node_embeddings(nodes, embedding_provider, embedding_model)
            return nodes
        except Exception:
            return []

    def extract_edges(self, document_content: str, nodes: List[Node], system_prompt: Optional[str] = None, user_prompt_template: Optional[str] = None, response_schema: Optional[dict] = None, max_tries: int = 5, **kwargs) -> List[Edge]:
        if not nodes:
            return []
        if not system_prompt or not user_prompt_template:
            return []

        nodes_str = "\n".join([f"- {node.properties.get('text')} ({node.properties.get('description')})" for node in nodes])
        user_prompt = user_prompt_template.format(text=document_content, nodes=nodes_str)

        try:
            response_message = self._call_llm(user_prompt=user_prompt, system_prompt=system_prompt, response_schema=response_schema, max_tries=max_tries)
            return self._parse_edges_from_json(response_message)
        except Exception:
            return []

    def extract_graph(self, document_content: str, **kwargs) -> Graph:
        nodes = self.extract_nodes(document_content, **kwargs)
        edges = self.extract_edges(document_content, nodes, **kwargs)
        return Graph(nodes=nodes, edges=edges)
