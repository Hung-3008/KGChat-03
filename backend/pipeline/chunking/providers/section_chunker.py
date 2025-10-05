import logging
import re
import uuid
from typing import List, Dict, Any, Optional

from backend.pipeline.chunking.base import BaseChunker
from backend.utils.text_processing import count_tokens

logger = logging.getLogger(__name__)


class SectionChunker(BaseChunker):
    """
    Creates chunks from structured data where each section becomes a chunk.
    Designed for PMC data format.
    """

    def __init__(self, min_chunk_tokens: int = 100):
        super().__init__(min_chunk_tokens=min_chunk_tokens)

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on line breaks.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        processed_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) > 100:
                processed_paragraphs.append(para.strip())
            else:
                sub_paras = re.split(r'\n', para)
                sub_paras = [sp.strip() for sp in sub_paras if sp.strip()]
                processed_paragraphs.extend(sub_paras)
        return [p.strip() for p in processed_paragraphs if p.strip() and len(p.strip()) > 10]

    def chunk(
        self,
        data: Dict[str, Any],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from PMC data where each section becomes a chunk.

        Args:
            data: PMC JSON data with title, metadata, and content_sections.
            document_metadata: Additional metadata to include with chunks.

        Returns:
            List of chunks with metadata, one per section.
        """
        pmc_data = data
        document_id = document_metadata.get("document_id", str(uuid.uuid4())) if document_metadata else str(uuid.uuid4())
        knowledge_level = document_metadata.get("knowledge_level", 1) if document_metadata else 1

        chunks = []
        chunk_index = 0

        # Title chunk
        title = pmc_data.get('title', '')
        if title:
            title_content = f"# {title}\n\n"
            title_tokens = count_tokens(title_content)
            if title_tokens >= self.min_chunk_tokens:
                meta = (document_metadata.copy() if document_metadata else {})
                meta.update({
                    "chunk_index": chunk_index, "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id, "knowledge_level": knowledge_level,
                    "section_type": "title", "section_header": "Document Title"
                })
                chunks.append({
                    "content": title_content, "tokens": title_tokens,
                    "paragraph_count": 1, "metadata": meta
                })
                chunk_index += 1

        # Metadata chunk
        metadata = pmc_data.get('metadata', {})
        if metadata:
            metadata_content = "## Document Information\n"
            metadata_content += f"- Word count: {metadata.get('word_count', 'N/A')}\n"
            metadata_content += f"- Estimated reading time: {metadata.get('estimated_reading_time_minutes', 'N/A')} minutes\n\n"
            metadata_tokens = count_tokens(metadata_content)
            if metadata_tokens >= self.min_chunk_tokens:
                meta = (document_metadata.copy() if document_metadata else {})
                meta.update({
                    "chunk_index": chunk_index, "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id, "knowledge_level": knowledge_level,
                    "section_type": "metadata", "section_header": "Document Information"
                })
                chunks.append({
                    "content": metadata_content, "tokens": metadata_tokens,
                    "paragraph_count": 1, "metadata": meta
                })
                chunk_index += 1

        # Content sections
        for section in pmc_data.get('content_sections', []):
            header = section.get('header', '')
            content = section.get('content', '')
            if header and content:
                section_content = f"## {header}\n\n{content}\n\n"
                section_tokens = count_tokens(section_content)
                if section_tokens < self.min_chunk_tokens:
                    continue

                meta = (document_metadata.copy() if document_metadata else {})
                meta.update({
                    "chunk_index": chunk_index, "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id, "knowledge_level": knowledge_level,
                    "section_type": "content", "section_header": header
                })
                chunks.append({
                    "content": section_content, "tokens": section_tokens,
                    "paragraph_count": len(self._split_into_paragraphs(section_content)),
                    "metadata": meta
                })
                chunk_index += 1

        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)

        logger.info(f"Created {len(chunks)} section-based chunks for document {document_id} (Level {knowledge_level}).")
        return chunks
