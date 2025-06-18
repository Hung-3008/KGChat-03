from typing import Dict, Type, TypeVar, Generic, Optional, Any, List
import logging
from ..utils.exceptions import ClientNotFoundError, ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')
C = TypeVar('C')


class BaseFactory(Generic[T, C]):
    def __init__(self):
        self._clients: Dict[str, Type[T]] = {}
        self._configs: Dict[str, Type[C]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, config_class: Optional[Type[C]] = None, 
                metadata: Optional[Dict[str, Any]] = None):

        def decorator(client_class: Type[T]) -> Type[T]:
            self.register_client(name, client_class, config_class, metadata)
            return client_class
        return decorator
    
    def register_client(self, name: str, client_class: Type[T], 
                       config_class: Optional[Type[C]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:

        name = name.strip().lower()
        
        self._clients[name] = client_class
        if config_class:
            self._configs[name] = config_class
        self._metadata[name] = metadata or {}
        
        logger.info(f"Registered client '{name}'")
    
    def create_client(self, name: str, config: Optional[C] = None, **kwargs) -> T:

        name = name.strip().lower()
        
        if name not in self._clients:
            available = ", ".join(self._clients.keys())
            raise ClientNotFoundError(
                f"Client '{name}' not found. Available: {available}"
            )
        
        client_class = self._clients[name]
        
        if config is None:
            if name in self._configs:
                config_class = self._configs[name]
                config = config_class()
            else:
                raise ConfigurationError(
                    f"No configuration provided for '{name}'"
                )
        
        return client_class(config, **kwargs)
    
    def get_registered_clients(self) -> List[str]:
        return list(self._clients.keys())
    
    def is_registered(self, name: str) -> bool:
        return name.strip().lower() in self._clients