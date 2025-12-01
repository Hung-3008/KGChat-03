from .base import BaseLLMClient, BaseLLMConfig, LLMResponse
from .factory import llm_registry

def initialize():
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
    "llm_registry"
]