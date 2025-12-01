"""
Keyword Extraction Module

This module extracts high-level and low-level keywords from user queries and
conversation history using the LLM. These keywords are used in the knowledge
graph retrieval pipeline to improve query accuracy.
"""
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import asyncio
from pydantic import BaseModel, Field

from backend.pipeline_prompts import PROMPTS
from backend.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)


class KeywordExtraction(BaseModel):
    """Pydantic model for structured keyword extraction output."""
    high_level_keywords: List[str] = Field(
        description="Overarching concepts or themes from the query"
    )
    low_level_keywords: List[str] = Field(
        description="Specific entities, details, or concrete terms from the query"
    )


async def extract_keywords(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    llm_client=None
) -> Tuple[List[str], List[str]]:

    if not query or not query.strip():
        logger.warning("Empty query provided for keyword extraction")
        return [], []

    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}"
            # Only use last 3 messages for context
            for msg in conversation_history[-3:]
        ])

    examples = "\n".join(PROMPTS.get("keywords_extraction_examples", []))

    extraction_prompt = PROMPTS.get("keywords_extraction", "").format(
        query=query,
        history=history_text,
        examples=examples
    )

    try:
        # Convert Pydantic model to JSON schema for Gemini
        from pydantic import TypeAdapter
        schema = TypeAdapter(KeywordExtraction).json_schema()

        response = llm_client.generate(
            user_prompt=extraction_prompt,
            response_schema=schema
        )

        # Parse response - LLMResponse has message field
        if hasattr(response, 'message'):
            content = response.message
        elif isinstance(response, dict):
            content = response.get("message", "")
        else:
            content = str(response)

        # Parse JSON from string
        if isinstance(content, str):
            try:
                keywords_dict = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response: {content}")
                return [], []
        else:
            keywords_dict = content

        # Create KeywordExtraction object from parsed data
        keywords_data = KeywordExtraction(**keywords_dict)
        high_level_keywords = keywords_data.high_level_keywords
        low_level_keywords = keywords_data.low_level_keywords

        return high_level_keywords, low_level_keywords

    except Exception as e:
        logger.error(f"Error during keyword extraction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], []
