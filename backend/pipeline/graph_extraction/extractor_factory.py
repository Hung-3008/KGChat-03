import os
from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.providers.gemini_graph_extractor import GeminiGraphExtractor
from backend.pipeline.graph_extraction.providers.ollama_graph_extractor import OllamaGraphExtractor


class GraphExtractorFactory:
    _MAP = {
        "gemini": GeminiGraphExtractor,
        "ollama": OllamaGraphExtractor,
    }

    @staticmethod
    def create_graph_extractor(model_platform: str, model_name: str, **kwargs) -> GraphExtractorBase:
        factory_kwargs = {"model_name": model_name}
        factory_kwargs.update(kwargs)
        cls = GraphExtractorFactory._MAP.get(model_platform)
        if not cls:
            raise ValueError(f"Unsupported model platform: {model_platform}")
        return cls(**factory_kwargs)
