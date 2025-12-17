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
        Merge to keep chunk count small and reduce LLM calls.
        - If more than 3 chunks, merge small ones into the next until token ~1800.
        - Target: <= 3 chunks to limit per-file LLM traffic.
        """
        if len(chunks) <= 3:
            return chunks

        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return chunks

        merged_chunks: List[str] = []
        current_chunk = ""

        def token_len(text: str) -> int:
            return len(enc.encode(text)) if text else 0

        for chunk in chunks:
            if not current_chunk:
                current_chunk = chunk
                continue

            if token_len(current_chunk) < 1800:
                current_chunk += "\n\n" + chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)

        # If still too many, greedily merge last ones
        while len(merged_chunks) > 3:
            last = merged_chunks.pop()
            merged_chunks[-1] = merged_chunks[-1] + "\n\n" + last

        return merged_chunks


__all__ = ["SectionChunker"]
