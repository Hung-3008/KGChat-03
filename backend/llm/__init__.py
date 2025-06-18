from .base import BaseLLMClient, BaseLLMConfig, LLMResponse
from .factory import llm_registry
from .providers.gemini import GeminiClient, GeminiConfig

def initialize():
    """Initialize the LLM system by registering all providers"""
    try:
        llm_registry.register_all_providers()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to auto-register providers: {e}")


initialize()

__all__ = [
    "BaseLLMClient",
    "BaseLLMConfig", 
    "LLMResponse",
    "llm_registry",
    "GeminiClient",
    "GeminiConfig"
]