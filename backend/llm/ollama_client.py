from ollama import chat
from typing import Optional, Dict, Union
from pydantic import BaseModel


class OllamaClient:
    def __init__(self, config: dict, **kwargs):
        self.model = config.get("model", "llama3.1:1b")
        self.temperature = config.get("temperature", 0.85)
        self.top_p = config.get("top_p", 0.9)
    def _normal_response(self, prompt: str) -> str:
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    def _structured_response(self, prompt: str, format: BaseModel) -> Dict:
        
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=format.model_json_schema()
        )
        return dict(format.model_validate_json(response.message.content))

    def generate(self, prompt: str, format: Optional[BaseModel] = None) -> Union[str, Dict]:
        if format:
            return self._structured_response(prompt, format)
        else:
            return self._normal_response(prompt)