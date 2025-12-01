from typing import Dict, Any, Optional, List
from ..base.llm_config import BaseLLMConfig


def build_generation_config(config: BaseLLMConfig, field_map: Optional[Dict[str, str]] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
    field_map = field_map or {}
    default_attrs = [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "response_mime_type",
    ]
    attrs = include if include is not None else default_attrs
    payload: Dict[str, Any] = {}
    for attr in attrs:
        val = getattr(config, attr, None)
        if val is None:
            continue
        key = field_map.get(attr, attr)
        payload[key] = val
    return payload
