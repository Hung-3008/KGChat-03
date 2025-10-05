from backend.pipeline.chunking.base import BaseChunker
from backend.pipeline.chunking.providers.recursive_text_chunker import RecursiveTextChunker
from backend.pipeline.chunking.providers.section_chunker import SectionChunker


class ChunkerFactory:
    """Factory for creating chunker instances."""

    @staticmethod
    def create_chunker(chunker_type: str, **kwargs) -> BaseChunker:
        """
        Creates a chunker of a specific type.

        Args:
            chunker_type: The type of chunker to create ('recursive_text' or 'section').
            **kwargs: Arguments to pass to the chunker's constructor.

        Returns:
            An instance of a BaseChunker implementation.

        Raises:
            ValueError: If an unsupported chunker_type is provided.
        """
        if chunker_type == "recursive_text":
            return RecursiveTextChunker(**kwargs)
        elif chunker_type == "section":
            return SectionChunker(**kwargs)
        else:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
