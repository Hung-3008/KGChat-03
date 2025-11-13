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
        return out


__all__ = ["SectionChunker"]
