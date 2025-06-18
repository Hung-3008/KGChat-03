import google.generativeai as genai
from typing import Optional, Dict, Any
import logging
from ...base.llm_client import BaseLLMClient, LLMResponse
from ...utils.exceptions import AuthenticationError
from .gemini_config import GeminiConfig

logger = logging.getLogger(__name__)

class GeminiClient(BaseLLMClient):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.config: GeminiConfig = config
        self._model = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        if not self.config.api_key:
            raise AuthenticationError("Gemini API key is required")
        
        genai.configure(api_key=self.config.api_key)
        generation_config = self.config.get_generation_config()
        
        self._model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=generation_config
        )
        
        logger.info(f"Initialized Gemini client with model: {self.config.model_name}")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        try:
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            else:
                full_prompt = user_prompt
            
            response = self._model.generate_content(full_prompt)
            
            message = response.text if response.parts else "No response generated"
            
            metadata = {
                "model": self.config.model_name,
                "finish_reason": getattr(response, 'finish_reason', None),
                "usage": getattr(response, 'usage_metadata', None)
            }
            
            return LLMResponse(message=message, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise