import yaml
import os
from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.providers.gemini_graph_extractor import GeminiGraphExtractor
from backend.pipeline.graph_extraction.providers.ollama_graph_extractor import OllamaGraphExtractor


class GraphExtractorFactory:

    @staticmethod
    def create_graph_extractor(model_platform: str, model_name:str, **kwargs) -> GraphExtractorBase:
        """Creates a graph extractor for the specified model platform."""
        
        factory_kwargs = {}
        factory_kwargs['model_name'] = model_name
        factory_kwargs.update(kwargs)

        if model_platform == "gemini":
            return GeminiGraphExtractor(**factory_kwargs)
        elif model_platform == "ollama":
            return OllamaGraphExtractor(**factory_kwargs)
        # elif model_platform == "openai":
        #     return OpenAIGraphExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported model platform: {model_platform}")
