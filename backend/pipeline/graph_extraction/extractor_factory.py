import yaml
import os
from backend.pipeline.graph_extraction.base import GraphExtractorBase
from backend.pipeline.graph_extraction.providers.gemini_graph_extractor import GeminiGraphExtractor
from backend.pipeline.graph_extraction.providers.ollama_graph_extractor import OllamaGraphExtractor


class GraphExtractorFactory:
    """Factory for creating graph extractor instances."""

    @staticmethod
    def create_graph_extractor(model_platform: str, model_name:str, config_file_path: str, **kwargs) -> GraphExtractorBase:
        """Creates a graph extractor for the specified model platform."""
        
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                config = yaml.safe_load(f).get("graph_extraction", {})

        #print("=== Config file: ", config_file_path)
        
        factory_kwargs = {}
        factory_kwargs['model_name'] = model_name

        #print("=== Model name: ", model_name)
        
        prompt_config = config.get("prompt_settings", {})
        if prompt_config:
            base_prompt_path = prompt_config.get("base_path")
            #print("=== Base prompt path: ", base_prompt_path)
            factory_kwargs['prompt_settings'] = {
                "base_path": base_prompt_path,
                "stages": prompt_config.get("stages", {})
            }

        factory_kwargs.update(kwargs)
        #print("=== Model platform: ", model_platform)
        if model_platform == "gemini":
            return GeminiGraphExtractor(**factory_kwargs)
        elif model_platform == "ollama":
            return OllamaGraphExtractor(**factory_kwargs)
        # elif model_platform == "openai":
        #     return OpenAIGraphExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported model platform: {model_platform}")
