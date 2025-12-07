import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
root_path = Path(__file__).parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))

try:
    from backend.llm.providers.gemini.gemini_client import GeminiClient
    from backend.llm.providers.gemini.gemini_config import GeminiConfig
    from backend.pipeline.keyword_extractor import extract_keywords
    print("Imported modules successfully")
except ImportError as e:
