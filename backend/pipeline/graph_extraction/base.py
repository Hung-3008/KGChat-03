from abc import ABC, abstractmethod
from typing import List
from .graph_elements import Node, Edge, Graph


class GraphExtractorBase(ABC):
    """Abstract base class for graph extraction."""

    @abstractmethod
    def extract_nodes(self, document_content: str, **kwargs) -> List[Node]:
        """Extracts nodes from the given document content."""
        pass

    @abstractmethod
    def extract_edges(self, document_content: str, nodes: List[Node], **kwargs) -> List[Edge]:
        """Extracts edges from the given document content."""
        pass

    def extract_graph(self, document_content: str) -> Graph:
        """Extracts a graph from the given document content."""
        nodes = self.extract_nodes(document_content)
        edges = self.extract_edges(document_content, nodes)
        return Graph(nodes=nodes, edges=edges)
