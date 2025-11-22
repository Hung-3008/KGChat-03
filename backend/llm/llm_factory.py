from typing import Optional, Dict
from pydantic import BaseModel


class LLMFactory:
    @staticmethod
    def create_client(config: dict):
        client_name = config.get("client", "ollama")
        if client_name == "ollama":
            from .ollama_client import OllamaClient

            return OllamaClient(config)
        elif client_name == "gemini":
            from .gemini_client import GeminiClient
            
            return GeminiClient(config)
        else:
            raise ValueError(f"Unsupported LLM client: {client_name}")