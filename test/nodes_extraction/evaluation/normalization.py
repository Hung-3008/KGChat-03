import unicodedata
import re

# Simple abbreviation mapping (can be expanded)
ABBREVIATION_MAP = {
    "WD": "Wilson disease",
    "FAP": "familial adenomatous polyposis",
    "CT": "copper toxicosis",
    "APC": "adenomatous polyposis coli"
}

def normalize_text(text: str) -> str:
    """Applies basic text normalization.
    - Lowercase
    - Trim whitespace
    - Unify punctuation (hyphens)
    - Unicode NFKC normalization
    """
    text = text.lower().strip()
    text = re.sub(r'[\u2010-\u2015]', '-', text)  # Unify hyphens
    return unicodedata.normalize('NFKC', text)

def expand_abbreviations(text: str) -> str:
    """Expands known medical abbreviations.
    Note: This is a simple implementation. A real-world scenario would need a comprehensive dictionary.
    """
    # Use word boundaries to avoid replacing parts of words
    for abbr, expansion in ABBREVIATION_MAP.items():
        text = re.sub(r'\b' + re.escape(abbr.lower()) + r'\b', expansion.lower(), text)
    return text

def normalize_gene_casing(text: str, entity_type: str) -> str:
    """Normalizes gene/protein names to uppercase if the type is 'Modifier'."""
    if entity_type == 'Modifier':
        return text.upper()
    return text

def get_normalized_entity(entity: dict) -> dict:
    """Returns a new entity dictionary with normalized text and type."""
    text = entity.get('text', '')
    entity_type = entity.get('type', '')

    # 1. Basic normalization
    normalized_text = normalize_text(text)

    # 2. Medical normalization
    normalized_text = expand_abbreviations(normalized_text)
    normalized_text = normalize_gene_casing(normalized_text, entity_type)

    return {
        "text": normalized_text,
        "type": entity_type,
        "original_text": text
    }

def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Removes duplicate entities based on normalized text and type."""
    seen = set()
    deduplicated = []
    for entity in entities:
        # Create a unique key for each entity based on its normalized form
        key = (entity['text'], entity['type'])
        if key not in seen:
            seen.add(key)
            deduplicated.append(entity)
    return deduplicated
