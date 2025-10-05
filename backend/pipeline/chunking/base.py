from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseChunker(ABC):
    """Abstract base class for document chunking."""

    def __init__(
        self,
        min_chunk_tokens: int = 100,
    ):
        """
        Initialize the base chunker.

        Args:
            min_chunk_tokens: Minimum number of tokens for a valid chunk.
        """
        self.min_chunk_tokens = min_chunk_tokens

    @abstractmethod
    def chunk(
        self,
        data: Any,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Splits a document into chunks.

        Args:
            data: The document content to chunk (e.g., text or structured data).
            document_metadata: Optional metadata for the document.

        Returns:
            A list of chunk dictionaries.
        """
        pass
