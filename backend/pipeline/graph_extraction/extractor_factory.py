from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.providers.gemini_graph_extractor import GeminiGraphExtractor
from backend.pipeline.graph_extraction.providers.ollama_graph_extractor import OllamaGraphExtractor


class GraphExtractorFactory:
    """Factory for creating graph extractor instances."""

    @staticmethod
    def create_graph_extractor(model_platform: str, **kwargs) -> GraphExtractorBase:
        """Creates a graph extractor for the specified model platform."""
        if model_platform == "gemini":
            return GeminiGraphExtractor(**kwargs)
        elif model_platform == "ollama":
            return OllamaGraphExtractor(**kwargs)
        # In the future, you can add other implementations here:
        # elif model_platform == "openai":
        #     return OpenAIGraphExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported model platform: {model_platform}")
