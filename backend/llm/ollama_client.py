from ollama import chat
from typing import Optional, Dict, Union
from pydantic import BaseModel


class OllamaClient:
    def __init__(self, config: dict, **kwargs):
        self.model = config.get("model", "llama3.1:1b")
        self.temperature = config.get("temperature", 0.85)
        self.top_p = config.get("top_p", 0.9)
        # optional seed for reproducibility if supported by Ollama
        self.seed = config.get("seed", None)
    def _normal_response(self, prompt: str) -> str:
        options = {"temperature": self.temperature, "top_p": self.top_p}
        if self.seed is not None:
            options["seed"] = self.seed

        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        return response.message.content

    def _structured_response(self, prompt: str, format: BaseModel) -> Dict:
        options = {"temperature": self.temperature, "top_p": self.top_p}
        if self.seed is not None:
            options["seed"] = self.seed

        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=format.model_json_schema(),
            options=options,
        )
        return dict(format.model_validate_json(response.message.content))

    def generate(self, prompt: str, format: Optional[BaseModel] = None) -> Union[str, Dict]:
        if format:
            return self._structured_response(prompt, format)
        else:
            return self._normal_response(prompt)