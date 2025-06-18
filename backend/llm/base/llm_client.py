from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
from pydantic import BaseModel
from .llm_config import BaseLLMConfig

class LLMResponse(BaseModel):
    message: Union[str, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMClient(ABC):    
    def __init__(self, config: BaseLLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        pass