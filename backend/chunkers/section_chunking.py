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

    def _merge_chunks(self, chunks: List[str], max_chunks: int = 4) -> List[str]:
        """
        Adaptive chunking algorithm: creates balanced chunks of similar token counts.
        Prevents chunks from being too long or too short.
        
        Args:
            chunks: List of section texts
            max_chunks: Maximum number of output chunks (default 4)
        
        Returns:
            List of merged chunks, evenly distributed by token count
        """
        if len(chunks) <= max_chunks:
            return chunks

        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # Fallback: use character count / 4 as rough token estimate
            enc = None

        # Calculate token counts
        if enc:
            chunk_tokens = [len(enc.encode(c)) for c in chunks]
        else:
            chunk_tokens = [len(c) // 4 for c in chunks]
        
        total_tokens = sum(chunk_tokens)
        target_per_chunk = total_tokens // max_chunks
        
        # Adaptive merging: create balanced chunks
        merged = []
        current = ""
        current_tokens = 0
        
        for i, chunk in enumerate(chunks):
            tokens = chunk_tokens[i]
            
            if current_tokens == 0:
                # Start new chunk
                current = chunk
                current_tokens = tokens
            elif current_tokens + tokens <= target_per_chunk * 1.5:
                # Merge if within 150% of target (allows some flexibility)
                current += "\n\n" + chunk
                current_tokens += tokens
            else:
                # Current chunk is good, save it
                merged.append(current)
                current = chunk
                current_tokens = tokens
                
                # Stop if we've created max_chunks-1 (last one gets remainder)
                if len(merged) >= max_chunks - 1:
                    break
        
        # Handle remaining chunks
        if current:
            if len(merged) < max_chunks:
                # Add as separate chunk
                merged.append(current)
            else:
                # Merge with last chunk to avoid exceeding max_chunks
                merged[-1] += "\n\n" + current
        
        # Merge any remaining unprocessed chunks into last chunk
        if i + 1 < len(chunks):
            remaining = "\n\n".join(chunks[i+1:])
            if remaining:
                merged[-1] += "\n\n" + remaining
        
        return merged


__all__ = ["SectionChunker"]
