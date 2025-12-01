import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from ..utils.exceptions import ClientNotFoundError

logger = logging.getLogger(__name__)

class LLMRegistry:
    def __init__(self, config_path: Optional[str] = None):
        # default to centralized config location
        self.config_path = config_path or "backend/configs/llm_configs.yml"
        self.config_data: Dict[str, Any] = {}
        self._clients: Dict[str, type] = {}
        self._configs: Dict[str, type] = {}
        self._defaults: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self) -> None:
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                # no config found; leave config_data empty
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                content = self._substitute_env_vars(content)
                self.config_data = yaml.safe_load(content) or {}
            
            # config loaded
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config_data = {}
    
    def _substitute_env_vars(self, content: str) -> str:
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, f"${{{var_name}}}")
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    
    def _import_class(self, class_path: str):
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            logger.error(f"Failed to import {class_path}: {e}")
            raise ImportError(f"Cannot import {class_path}: {e}")
    
    def register_all_providers(self) -> None:
        if not self.config_data:
            self.load_config()
        
        providers = self.config_data.get('providers', {})
        
        for provider_name, provider_config in providers.items():
            try:
                self._register_provider(provider_name, provider_config)
            except Exception:
                # skip faulty providers
                continue
    
    def _register_provider(self, name: str, config: Dict[str, Any]) -> None:
        client_path = config.get('client')
        config_path = config.get('config')
        
        if not client_path:
            raise ValueError(f"No client class for provider '{name}'")
        
        client_class = self._import_class(client_path)
        config_class = self._import_class(config_path) if config_path else None

        self._clients[name] = client_class
        self._configs[name] = config_class
        self._defaults[name] = config.get('defaults', {})
    
    def get_default_provider(self) -> Optional[str]:
        return self.config_data.get('default_provider')
    
    def create_llm_client(self, provider_name: Optional[str] = None, **overrides):

        if not provider_name:
            provider_name = self.get_default_provider()
        
        if not provider_name:
            raise ValueError("No provider specified and no default provider")
        
        if provider_name not in self._clients:
            available = ", ".join(self._clients.keys())
            raise ClientNotFoundError(
                f"Client '{provider_name}' not found. Available: {available}"
            )
        
        client_class = self._clients[provider_name]
        config_class = self._configs[provider_name]
        
        if not config_class:
            raise ValueError(f"No configuration class found for '{provider_name}'")
        
        defaults = self._defaults.get(provider_name, {})
        final_config = {**defaults, **overrides}
        
        config = config_class(**final_config)
        return client_class(config)

    def embed_texts(self, texts: List[str], provider_name: Optional[str] = None, model: Optional[str] = None, max_tries: int = 3, **overrides) -> List[List[float]]:
        """Convenience helper to generate embeddings using a provider's client.

        - texts: list of input strings
        - provider_name: optional provider (defaults to registry default)
        - model: optional embedding model override
        - max_tries: passed to provider client if supported
        - overrides: other config overrides passed when constructing the client
        """
        if not provider_name:
            provider_name = self.get_default_provider()

        client = self.create_llm_client(provider_name=provider_name, **overrides)

        # Prefer a batch embed API if available
        if hasattr(client, 'embed'):
            try:
                # many providers accept (texts, model=..., max_tries=...)
                return client.embed(texts, model=model, max_tries=max_tries)
            except TypeError:
                # fallback: try without named args
                return client.embed(texts)

        # If client only provides single-text embed (e.g., embed(text)), call per-text
        if hasattr(client, 'embed_single'):
            embeddings = []
            for t in texts:
                embeddings.append(client.embed_single(t, model=model, max_tries=max_tries))
            return embeddings

        raise NotImplementedError(f"Provider '{provider_name}' does not implement embedding API")
    
    def get_available_providers(self) -> list:
        return list(self._clients.keys())

llm_registry = LLMRegistry()
