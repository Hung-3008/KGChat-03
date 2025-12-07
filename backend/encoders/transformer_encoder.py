from __future__ import annotations

from typing import List, Optional, Union
import asyncio

import torch
from transformers import AutoModel, AutoTokenizer


class TransformerEncoder:
    def __init__(
        self,
        model_name: Optional[str] = "NeuML/pubmedbert-base-embeddings",
        device: Optional[str] = "cpu",
        target_dimension: Optional[int] = None
    ):
        self.model_name = model_name
        self.device_str = device or "cpu"
        self.device = torch.device(self.device_str if (
            self.device_str == "cpu" or torch.cuda.is_available()) else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self.target_dimension = target_dimension

    def _ensure_model_loaded(self) -> None:
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            try:
                self._model.to(self.device)
            except Exception:
                pass

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> torch.Tensor:
        if not isinstance(texts, (list, tuple)):
            raise ValueError("`texts` must be a list of strings")
        if len(texts) == 0:
            raise ValueError("`texts` must not be empty")

        self._ensure_model_loaded()
        assert self._tokenizer is not None and self._model is not None

        all_embs: List[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            encoded = self._tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) if isinstance(
                v, torch.Tensor) else v for k, v in encoded.items()}

            with torch.no_grad():
                model_output = self._model(**encoded)

            last_hidden = model_output.last_hidden_state
            att_mask = encoded["attention_mask"]
            pooled = self._mean_pooling(last_hidden, att_mask)

            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embs.append(pooled.cpu())

        result = torch.cat(all_embs, dim=0)
        return result

    def embed_to_numpy(self, texts: List[str], **kwargs):
        import numpy as np

        tensor = self.embed(texts, **kwargs)
        return tensor.numpy()

    def _resize_embedding(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        Resize embedding to target dimension.

        Args:
            embedding: Embedding vector as list
            target_dim: Target dimension

        Returns:
            Resized embedding vector
        """
        if len(embedding) == target_dim:
            return embedding
        elif len(embedding) > target_dim:
            return embedding[:target_dim]
        else:
            return embedding + [0.0] * (target_dim - len(embedding))

    async def embed_async(
        self,
        texts: Union[str, List[str]],
        target_dimension: Optional[int] = None,
        batch_size: int = 32,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Async method to generate embeddings.

        Args:
            texts: Single string or list of strings to embed
            target_dimension: Target dimension for embeddings (uses self.target_dimension if None)
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors as lists of floats
        """
        import numpy as np

        # Handle string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Use target_dimension from parameter or instance
        target_dim = target_dimension or self.target_dimension

        try:
            # Run embedding in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            embeddings_tensor = await loop.run_in_executor(
                None,
                lambda: self.embed_to_numpy(
                    texts, batch_size=batch_size, normalize=normalize)
            )

            # Convert numpy array to list of lists
            if isinstance(embeddings_tensor, np.ndarray):
                embeddings = embeddings_tensor.tolist()
            else:
                # If it's a torch.Tensor, convert to numpy then to list
                if hasattr(embeddings_tensor, 'numpy'):
                    embeddings = embeddings_tensor.numpy().tolist()
                else:
                    embeddings = list(embeddings_tensor)

            if embeddings and len(embeddings) > 0:
                first_dim = len(embeddings[0]) if embeddings[0] else 0

                if first_dim == 0:
                    # Return dummy embeddings if empty
                    return [[0.0] * (target_dim or 768) for _ in texts]

                # Resize embeddings if target_dimension is specified
                if target_dim and first_dim != target_dim:
                    embeddings = [
                        self._resize_embedding(emb, target_dim)
                        for emb in embeddings
                    ]

            else:
                # Return dummy embeddings if no embeddings received
                return [[0.0] * (target_dim or 768) for _ in texts]

            return embeddings

        except Exception:
            # Return dummy embeddings on error
            return [[0.0] * (target_dim or 768) for _ in texts]


__all__ = ["TransformerEncoder"]
