from pathlib import Path
from typing import List, Union
import json


class SectionChunker:
    def chunk(self, data: Union[dict, str, Path]) -> List[str]:
        if isinstance(data, (str, Path)):
            try:
                p = Path(data)
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return []
        if not isinstance(data, dict):
            return []
        sections = data.get("content_sections") or []
        out: List[str] = []
        for s in sections:
            c = s.get("content") if isinstance(s, dict) else None
            if c and isinstance(c, str) and c.strip():
                out.append(c.strip())
        
        return self._merge_chunks(out)

    def _merge_chunks(self, chunks: List[str], chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
        """
        Chunks text using LangChain's RecursiveCharacterTextSplitter with tiktoken encoder.
        
        Args:
            chunks: List of section texts (will be joined first)
            chunk_size: Target token count per chunk (default 1500)
            chunk_overlap: Overlap in tokens (default 200)
            
        Returns:
            List of text chunks
        """
        if not chunks:
            return []

        # Join all sections with double newlines to preserve structure
        full_text = "\n\n".join(chunks)
        
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4", # Use cl100k_base encoding
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            return splitter.split_text(full_text)
            
        except ImportError:
            # Fallback if langchain is not installed (though it should be)
            # Simple character splitting
            import warnings
            warnings.warn("LangChain not found, falling back to simple character splitting.")
            return [full_text[i:i+4000] for i in range(0, len(full_text), 3800)] # Rough char approx

__all__ = ["SectionChunker"]
