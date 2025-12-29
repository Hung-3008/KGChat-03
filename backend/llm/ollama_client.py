from ollama import Client
from typing import Optional, Dict, Union, List
from pydantic import BaseModel
import threading
import itertools
import random

class OllamaClient:
    def __init__(self, config: dict, **kwargs):
        self.model = config.get("model", "llama3.1:1b")
        self.temperature = config.get("temperature", 0.85)
        self.top_p = config.get("top_p", 0.9)
        self.seed = config.get("seed", None)
        
        base_urls = config.get("base_url", "http://localhost:11434")
        if isinstance(base_urls, str):
            base_urls = [base_urls]
            
        self.clients = [Client(host=url) for url in base_urls]
        self.client_cycle = itertools.cycle(self.clients)
        self.lock = threading.Lock()
        
        print(f"DEBUG: OllamaClient initialized with {len(self.clients)} instances: {base_urls}")

    def _get_next_client(self):
        with self.lock:
            return next(self.client_cycle)

    def _normal_response(self, prompt: str) -> str:
        client = self._get_next_client()
        options = {"temperature": self.temperature, "top_p": self.top_p}
        if self.seed is not None:
            options["seed"] = self.seed

        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        return response.message.content

    def _structured_response(self, prompt: str, format: Union[BaseModel, Dict]) -> Dict:
        client = self._get_next_client()
        options = {"temperature": self.temperature, "top_p": self.top_p}
        if self.seed is not None:
            options["seed"] = self.seed

        if isinstance(format, dict):
            schema = format
        else:
            schema = format.model_json_schema()

        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            options=options,
        )
        
        if isinstance(format, dict):
            import json
            return json.loads(response.message.content)
        else:
            return dict(format.model_validate_json(response.message.content))

    def generate(self, prompt: str, format: Optional[Union[BaseModel, Dict]] = None) -> Union[str, Dict]:
        if format:
            return self._structured_response(prompt, format)
        else:
            return self._normal_response(prompt)