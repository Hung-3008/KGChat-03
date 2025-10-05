from abc import ABC, abstractmethod
from typing import List
from .graph_elements import Node


class GraphExtractorBase(ABC):
    """Abstract base class for graph extraction."""

    @abstractmethod
    def extract_nodes(self, document_content: str) -> List[Node]:
        """Extracts nodes from the given document content."""
        pass
