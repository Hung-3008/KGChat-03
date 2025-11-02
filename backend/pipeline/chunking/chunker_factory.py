from backend.pipeline.chunking.base import BaseChunker
from backend.pipeline.chunking.providers.recursive_text_chunker import RecursiveTextChunker
from backend.pipeline.chunking.providers.section_chunker import SectionChunker


class ChunkerFactory:
    _MAP = {
        "recursive_text": RecursiveTextChunker,
        "section": SectionChunker,
    }

    @staticmethod
    def create_chunker(chunker_type: str, **kwargs) -> BaseChunker:
        cls = ChunkerFactory._MAP.get(chunker_type)
        if not cls:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
        return cls(**kwargs)
