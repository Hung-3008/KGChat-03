import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.utils.logging import get_logger
from backend.pipeline_prompts import PROMPTS
logger = get_logger(__name__)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Convert text to a valid filename.

    Args:
        text: Text to convert
        max_length: Maximum length of the filename

    Returns:
        Sanitized filename
    """
    # Remove special symbols from text
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip('_')


def create_rag_prompt(query: str, formatted_text: str, conversation_history: Optional[str] = None) -> str:
    """
    Create RAG prompt ready to be passed to LLM.

    Args:
        query: User's query
        formatted_text: Text formatted from format_retrieval_results
        conversation_history: Conversation history (optional)

    Returns:
        Prompt string ready to be passed to LLM
    """
    # Use prompt template from pipeline_prompts.py
    prompt_template = PROMPTS.get("test_full_rag_prompt", "")

    # Format conversation history
    history_text = conversation_history if conversation_history else "(No previous conversation)"

    # Format prompt with values
    prompt = prompt_template.format(
        query=query,
        formatted_text=formatted_text,
        conversation_history=history_text
    )

    return prompt


def save_retrieval_result(
    query: str,
    formatted_text: str,
    result: dict,
    conversation_history: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Save formatted retrieval result to output_pipeline_retrie folder.
    Format according to RAG prompt structure for direct use with LLM.

    Args:
        query: Original query
        formatted_text: Text formatted from format_retrieval_results
        result: Dictionary containing all results
        conversation_history: Conversation history (optional)
        output_dir: Output directory (optional, defaults to output_pipeline_retrie at root)

    Returns:
        Path to saved file or None if error
    """
    try:
        # Import QueryIntent if available (optional dependency)
        query_intent_enum = None
        has_query_intent = False
        try:
            from backend.pipeline.query_analyzer import QueryIntent
            query_intent_enum = QueryIntent
            has_query_intent = True
        except ImportError:
            pass

        # Create output_pipeline_retrie folder if it doesn't exist
        if output_dir is None:
            # Get root path from current file (backend/retrieval/triple_level_retrieval.py)
            # Go up 2 levels to reach root: backend/retrieval -> backend -> root
            current_file = Path(__file__)
            root_path = current_file.parent.parent.parent
            output_dir = root_path / "output_pipeline_retrie"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Create filename based on query and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = sanitize_filename(query, max_length=50)
        filename = f"prompt_{timestamp}_{query_sanitized}.txt"
        file_path = output_dir / filename

        # Create RAG prompt ready to be passed to LLM
        rag_prompt = create_rag_prompt(
            query, formatted_text, conversation_history)

        # Create file content with prompt and metadata
        content_parts = []

        # Main prompt section (ready to copy to LLM)
        content_parts.append("=" * 80)
        content_parts.append(
            "PROMPT READY FOR LLM (Copy the section below to use)")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(rag_prompt)
        content_parts.append("")

        # Metadata section (for reference)
        content_parts.append("=" * 80)
        content_parts.append("METADATA (For reference only)")
        content_parts.append("=" * 80)
        content_parts.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Intent (if available)
        intent = result.get('intent')
        if intent:
            if has_query_intent and hasattr(intent, 'name'):
                content_parts.append(f"Intent: {intent.name}")
            else:
                content_parts.append(f"Intent: {intent}")
        else:
            content_parts.append("Intent: N/A")

        # Keywords (if available)
        high_level_keywords = result.get('high_level_keywords', [])
        low_level_keywords = result.get('low_level_keywords', [])

        # Display keywords if available (for HEALTHCARE_RELATED or any keywords)
        if high_level_keywords or low_level_keywords:
            # Only display if intent is HEALTHCARE_RELATED (if QueryIntent exists) or if QueryIntent doesn't exist
            should_show_keywords = True
            if has_query_intent and query_intent_enum and intent:
                should_show_keywords = (
                    intent == query_intent_enum.HEALTHCARE_RELATED)

            if should_show_keywords:
                content_parts.append(
                    f"High-level keywords: {', '.join(high_level_keywords)}")
                content_parts.append(
                    f"Low-level keywords: {', '.join(low_level_keywords)}")

        # Retrieval statistics
        if result.get('retrieval_result'):
            retrieval = result['retrieval_result']
            content_parts.append(f"Retrieval Statistics:")
            content_parts.append(
                f"  - Level 1 nodes: {len(retrieval.get('level1_nodes', []))}")
            content_parts.append(
                f"  - Level 2 nodes: {len(retrieval.get('level2_nodes', []))}")
            content_parts.append(
                f"  - Relationships: {len(retrieval.get('relationships', []))}")

        # Write file
        full_content = "\n".join(content_parts)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        logger.info(f"Saved retrieval result to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Error saving retrieval result: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None