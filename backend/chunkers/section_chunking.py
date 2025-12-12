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

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge small chunks if there are too many chunks.
        Rule: If chunks > 5, merge chunks with < 1000 tokens into the next chunk.
        """
        if len(chunks) <= 5:
            return chunks

        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # Fallback if tiktoken not installed, though it should be
            return chunks

        merged_chunks = []
        current_chunk = ""
        
        for i, chunk in enumerate(chunks):
            if not current_chunk:
                current_chunk = chunk
            else:
                # Check token count of current_chunk
                tokens = enc.encode(current_chunk)
                if len(tokens) < 1000:
                    # Merge with next chunk (current iteration)
                    current_chunk += "\n\n" + chunk
                else:
                    # Current chunk is big enough, append it and start new
                    merged_chunks.append(current_chunk)
                    current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
            
        return merged_chunks


__all__ = ["SectionChunker"]
