from typing import List, Dict, Optional, Tuple
import json
import ast

from backend.encoders.transformer_encoder import TransformerEncoder
from backend.node_extractor.supper_prompts import get_15_umls_prompts, get_filter_prompt
from backend.node_extractor.schema import TUI_TO_SEMTYPE


class NodeExtractor:
    """Extractor that runs the input through each of the 15 super-prompts organized by UMLS Semantic Groups.

    Behavior (2-stage pipeline):
    
    STAGE 1: DECOMPOSE & EXTRACT
    - For each super_prompt (1..15) call the LLM with the prompt populated with the input text.
    - Parse the LLM response expecting a JSON object mapping TUI -> list of entities
      (each entity should have `name`), but tolerant of variations.
    - Aggregate all found entities (may contain duplicates, misclassifications, hallucinations).
    
    STAGE 2: FILTER & CONSOLIDATE
    - Send all raw candidates + original text to a filter prompt.
    - The filter prompt validates each entity:
      1. Checks if entity name exists in original text (no hallucinations)
      2. Verifies semantic_type is correct for the entity (no misclassifications)
      3. Consolidates duplicates (picks best classification if entity appears multiple times)
    
    STAGE 3: COMPUTE EMBEDDINGS
    - Compute embeddings for names via the provided encoder.
    - Return a list of dicts containing: { name, semantic_type, name_embedding }.
    
    The 15 prompts cover:
    1. Activities & Behaviors
    2. Anatomy
    3. Chemicals & Drugs
    4. Concepts & Ideas
    5. Devices
    6. Disorders
    7. Genes & Molecular Sequences
    8. Geographic Areas
    9. Living Beings
    10. Objects
    11. Occupations
    12. Organizations
    13. Phenomena
    14. Physiology
    15. Procedures
    """

    def __init__(self, llm_client, model_name: str, embedding_model: str,
                 encoder: Optional[TransformerEncoder] = None, device: str = "cpu",
                 use_filter: bool = True):
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.encoder = encoder or TransformerEncoder(model_name=embedding_model, device=device)
        self.use_filter = use_filter  # Enable/disable filter stage

    def _safe_parse_json(self, text: str) -> Optional[object]:
        """Try to extract and parse a JSON object from LLM text output.

        Returns parsed object (dict/list) or None.
        """
        if not text or not text.strip():
            return None
        # If it's already a Python/JSON structure
        if isinstance(text, (dict, list)):
            return text

        # Try direct json.loads
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find the first/last braces and load that slice
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last != -1 and last > first:
            snippet = text[first:last+1]
            try:
                return json.loads(snippet)
            except Exception:
                try:
                    return ast.literal_eval(snippet)
                except Exception:
                    pass

        # Try to parse as a list (brackets)
        first_l = text.find('[')
        last_l = text.rfind(']')
        if first_l != -1 and last_l != -1 and last_l > first_l:
            snippet = text[first_l:last_l+1]
            try:
                return json.loads(snippet)
            except Exception:
                try:
                    return ast.literal_eval(snippet)
                except Exception:
                    pass

        return None

    def _extract_entities_from_parsed(self, parsed: object) -> List[Tuple[str, str]]:
        """Normalize parsed LLM output into list of (name, semantic_type) tuples.

        Support multiple shapes:
        - dict mapping TUI -> list of entities (each entity dict with name OR simple strings)
        - dict with key 'entities' -> list of entity dicts
        - plain list of entity dicts
        
        Map TUI codes to semantic_type names.
        """
        out: List[Tuple[str, str]] = []
        if parsed is None:
            return out

        # Case: dict with TUIs as keys
        if isinstance(parsed, dict):
            # If it has top-level 'entities' key
            if 'entities' in parsed and isinstance(parsed['entities'], list):
                items = parsed['entities']
                for it in items:
                    if not it:
                        continue
                    if isinstance(it, dict):
                        name = (it.get('name') or it.get('label') or '').strip()
                        sem = (it.get('semantic_type') or it.get('TUI') or it.get('tui') or it.get('type') or '').strip()
                        if name:
                            out.append((name, sem))
                return out

            # Otherwise assume keys are TUIs -> lists
            for key, val in parsed.items():
                if not isinstance(val, list):
                    continue
                # Get semantic type name from TUI mapping
                sem_type_name = TUI_TO_SEMTYPE.get(key.upper(), '') if key and key.upper().startswith('T') else ''
                
                for it in val:
                    if not it:
                        continue
                    if isinstance(it, dict):
                        # Entity is a dict with name field
                        name = (it.get('name') or it.get('label') or '').strip()
                        # Use TUI key's semantic type, or entity's own type if specified
                        sem = sem_type_name or (it.get('semantic_type') or it.get('TUI') or it.get('tui') or it.get('type') or '').strip()
                    elif isinstance(it, str):
                        # Entity is a simple string - use it as name
                        # This is the most common case from LLM output
                        name = it.strip()
                        sem = sem_type_name  # Use the TUI's semantic type
                    else:
                        continue
                    if name:
                        out.append((name, sem))
            return out

        # Case: list of entities
        if isinstance(parsed, list):
            for it in parsed:
                if not it:
                    continue
                if isinstance(it, dict):
                    name = (it.get('name') or it.get('label') or '').strip()
                    sem = (it.get('semantic_type') or it.get('TUI') or it.get('tui') or it.get('type') or '').strip()
                elif isinstance(it, str):
                    name = it.strip()
                    sem = ''
                else:
                    continue
                if name:
                    out.append((name, sem))
            return out

        return out

    def _apply_filter(self, original_text: str, raw_candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Apply filter prompt to validate and consolidate raw candidates.
        
        Args:
            original_text: The original input text
            raw_candidates: List of (name, semantic_type) tuples from stage 1
            
        Returns:
            Filtered and consolidated list of (name, semantic_type) tuples
        """
        if not raw_candidates:
            return []
        
        # Prepare raw candidates as JSON for the filter prompt
        raw_json = json.dumps([
            {"name": name, "semantic_type": sem}
            for name, sem in raw_candidates
        ], ensure_ascii=False, indent=2)
        
        # Get filter prompt and populate it
        filter_prompt_template = get_filter_prompt()
        filter_prompt = filter_prompt_template.replace('{insert_original_text_here}', original_text)
        filter_prompt = filter_prompt.replace('{insert_raw_extracted_json_here}', raw_json)
        
        try:
            # Call LLM with filter prompt
            filter_response = self.llm_client.generate(prompt=filter_prompt)
            
            # Parse the filtered response
            filtered_parsed = self._safe_parse_json(filter_response)
            
            if not filtered_parsed:
                # If filter fails, return original candidates (fallback)
                return raw_candidates
            
            # Extract filtered entities
            filtered_entities = []
            if isinstance(filtered_parsed, list):
                for item in filtered_parsed:
                    if isinstance(item, dict):
                        name = (item.get('name') or '').strip()
                        sem = (item.get('semantic_type') or '').strip()
                        if name:
                            filtered_entities.append((name, sem))
            elif isinstance(filtered_parsed, dict):
                # Maybe it's wrapped in a container
                if 'entities' in filtered_parsed:
                    items = filtered_parsed['entities']
                    for item in items:
                        if isinstance(item, dict):
                            name = (item.get('name') or '').strip()
                            sem = (item.get('semantic_type') or '').strip()
                            if name:
                                filtered_entities.append((name, sem))
            
            return filtered_entities if filtered_entities else raw_candidates
            
        except Exception as e:
            # If filter stage fails, return original candidates
            print(f"[WARNING] Filter stage failed: {e}. Using raw candidates.")
            return raw_candidates

    def extract(self, text: str) -> List[Dict]:
        """Run the input through 2-stage pipeline: (1) Extract from 15 prompts, (2) Filter & consolidate."""
        if not text or not text.strip():
            return []

        # ========== STAGE 1: DECOMPOSE & EXTRACT ==========
        # Get all 15 prompts organized by UMLS Semantic Groups
        all_prompts = get_15_umls_prompts()
        
        # Convert to list for iteration
        prompts = [
            all_prompts["group_1_activities"],
            all_prompts["group_2_anatomy"],
            all_prompts["group_3_chemicals"],
            all_prompts["group_4_concepts"],
            all_prompts["group_5_devices"],
            all_prompts["group_6_disorders"],
            all_prompts["group_7_genes"],
            all_prompts["group_8_geography"],
            all_prompts["group_9_living_beings"],
            all_prompts["group_10_objects"],
            all_prompts["group_11_occupations"],
            all_prompts["group_12_organizations"],
            all_prompts["group_13_phenomena"],
            all_prompts["group_14_physiology"],
            all_prompts["group_15_procedures"],
        ]

        found: List[Tuple[str, str]] = []

        for idx, tmpl in enumerate(prompts, start=1):
            # Replace the placeholder safely
            if '{insert your text here}' in tmpl:
                prompt = tmpl.replace('{insert your text here}', text)
            else:
                prompt = tmpl + "\n" + text

            try:
                resp = self.llm_client.generate(prompt=prompt)
            except Exception as e:
                # skip this prompt on failure
                continue

            parsed = self._safe_parse_json(resp)
            entities = self._extract_entities_from_parsed(parsed)
            
            # If nothing parsed and resp is already a dict-like object, try to use it
            if not entities and isinstance(resp, dict):
                entities = self._extract_entities_from_parsed(resp)

            for name, sem in entities:
                found.append((name, sem))

        if not found:
            return []

        # ========== STAGE 2: FILTER & CONSOLIDATE ==========
        if self.use_filter:
            # Apply filter prompt to validate and consolidate
            filtered = self._apply_filter(text, found)
            unique = filtered
        else:
            # Skip filter, just deduplicate
            seen = set()
            unique: List[Tuple[str, str]] = []
            for name, sem in found:
                key = (name.strip().lower(), (sem or '').strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                unique.append((name.strip(), (sem or '').strip()))

        if not unique:
            return []

        # ========== STAGE 3: COMPUTE EMBEDDINGS ==========
        names = [n for n, _ in unique]

        try:
            name_embs = self.encoder.embed_to_numpy(names).tolist()
        except Exception:
            # If embeddings fail, return basic structure without embeddings
            out = []
            for i, (n, s) in enumerate(unique):
                out.append({
                    "name": n,
                    "semantic_type": s,
                    "name_embedding": None,
                })
            return out

        out = []
        for i in range(len(unique)):
            n, s = unique[i]
            out.append({
                "name": n,
                "semantic_type": s,
                "name_embedding": name_embs[i] if i < len(name_embs) else None,
            })

        return out
