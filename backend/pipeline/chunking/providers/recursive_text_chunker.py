import logging
import uuid
from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.pipeline.chunking.base import BaseChunker
from backend.utils.text_processing import count_tokens

logger = logging.getLogger(__name__)


class RecursiveTextChunker(BaseChunker):
    """
    Implements document chunking using RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 12000,
        overlap_tokens: int = 1000,
        min_chunk_tokens: int = 100,
    ):
        """
        Initialize the chunker.

        Args:
            max_chunk_tokens: Maximum number of tokens per chunk.
            overlap_tokens: Number of tokens to overlap between chunks.
            min_chunk_tokens: Minimum number of tokens for a valid chunk.
        """
        super().__init__(min_chunk_tokens=min_chunk_tokens)
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

        # Estimate character lengths based on token counts (approx. 4 chars per token)
        self.max_chunk_size = max_chunk_tokens * 4
        self.overlap_size = overlap_tokens * 4

    def chunk(
        self,
        data: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create fixed-length chunks from a document with metadata using LangChain.

        Args:
            data: Raw document text.
            document_metadata: Additional metadata to include with chunks.

        Returns:
            List of chunks with metadata.
        """
        document_id = document_metadata.get("document_id", str(uuid.uuid4())) if document_metadata else str(uuid.uuid4())
        knowledge_level = document_metadata.get("knowledge_level", 1) if document_metadata else 1

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        raw_chunks = text_splitter.split_text(data)
        logger.info(f"Document split into {len(raw_chunks)} chunks using LangChain.")

        if not raw_chunks:
            return []

        chunks = []
        for i, chunk_content in enumerate(raw_chunks):
            chunk_tokens = count_tokens(chunk_content)

            if chunk_tokens < self.min_chunk_tokens:
                continue

            current_chunk_metadata = document_metadata.copy() if document_metadata else {}
            current_chunk_metadata.update({
                "chunk_index": i,
                "chunk_id": str(uuid.uuid4()),
                "total_chunks": len(raw_chunks),
                "document_id": document_id,
                "knowledge_level": knowledge_level,
            })

            chunk = {
                "content": chunk_content,
                "tokens": chunk_tokens,
                "metadata": current_chunk_metadata,
            }
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} fixed-length chunks for document {document_id} (Level {knowledge_level}).")
        return chunks
