from .embed_config import EMBEDDING_MODELS
from sentence_transformers import SentenceTransformer
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


class Embed:
    def __init__(self, model_key: str):
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f'model_key {model_key} not found in EMBEDDING_MODELS')
        model_name = EMBEDDING_MODELS[model_key]
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            self.model.eval()
            logger.info(f'Initialized {model_key} model')
            self.is_e5 = "e5" in model_name.lower()
        except Exception as e:
            logger.error(f'Failed to initialize {model_key} model: {e}')
            raise

    def embed(self, texts: Union[str, List[str]],) -> Union[List[float], List[List[float]]]:
        try:
            logger.debug(f'Generating embeddings')
            if isinstance(texts, str):
                text_list = [texts]
            else:
                text_list = texts
            embeddings = []
            if self.is_e5:
                text_list = [f"passage: {text}" for text in text_list]
            for text in text_list:
                embeddings.append(self.model.encode(text))
            if isinstance(texts, str):
                return embeddings[0] if embeddings else []
            return embeddings
        except Exception as e:
            logger.error(f'Failed to generate embeddings: {e}')
            raise

    



