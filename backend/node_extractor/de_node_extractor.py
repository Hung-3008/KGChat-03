from typing import List, Dict, Optional, Any
import json

from backend.encoders.transformer_encoder import TransformerEncoder
from backend.node_extractor.supper_prompts import get_4_cluster_prompts, get_filter_prompt
from backend.node_extractor.schema import (
    ActivityOutput, PhenomenonOutput, PhysicalObjectOutput, ConceptualEntityOutput, FilterOutput
)
from backend.node_extractor.hierarchical_graph import get_hierarchy


class NodeExtractor:
    """3-stage entity extraction pipeline:
    Stage 1: Decompose & Extract (implemented)
    Stage 2: Filter & Consolidate (implemented)
    Stage 3: Compute Embeddings (TODO)
    """

    def __init__(self, llm_client, model_name: str, embedding_model: str,
                 encoder: Optional[TransformerEncoder] = None, device: str = "cpu",
                 use_filter: bool = True):
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.encoder = encoder or TransformerEncoder(model_name=embedding_model, device=device)
        self.use_filter = use_filter
        self.cluster_prompts = get_4_cluster_prompts()
        self.schema_map = {
            "activity": ActivityOutput,
            "phenomenon": PhenomenonOutput,
            "physical_object": PhysicalObjectOutput,
            "conceptual_entity": ConceptualEntityOutput,
        }

    def extract(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Extract entities from text. Returns schema-based structured output."""
        if not text or not text.strip():
            return self._empty_output()
        
        # Stage 1: Decompose & Extract
        stage1_result = self._empty_output()
        for cluster_name, prompt_template in self.cluster_prompts.items():
            try:
                prompt = prompt_template.replace("{{TEXT}}", text)
                # call LLM for cluster
                resp = self.llm_client.generate(
                    prompt=prompt,
                    format=self.schema_map[cluster_name]
                )

                # Debug: show minimal info about raw response
                try:
                    resp_preview = resp if isinstance(resp, dict) else getattr(resp, '__dict__', str(resp))
                except Exception:
                    resp_preview = str(resp)
                print(f"[Stage1] cluster={cluster_name} raw_resp_type={type(resp)} preview_keys={list(resp_preview.keys()) if isinstance(resp_preview, dict) else 'non-dict'}")

                processed = self._process_response(resp, cluster_name)
                # Debug: counts per semantic field for this cluster
                counts = {k: len(v) for k, v in processed.items() if isinstance(v, list) and v}
                print(f"[Stage1] cluster={cluster_name} processed_counts={counts}")

                stage1_result[cluster_name] = processed
            except Exception as e:
                print(f"Error extracting {cluster_name}: {e}")
        
        # Stage 2: Filter & Consolidate
        if self.use_filter:
            return self._stage2_filter_consolidate(text, stage1_result)
        
        return stage1_result

    def _empty_output(self) -> Dict[str, Any]:
        return {k: v().dict() for k, v in self.schema_map.items()}

    def _process_response(self, response: Any, cluster: str) -> Dict[str, List[str]]:
        """Process LLM response (already in schema format) and deduplicate."""
        output = self.schema_map[cluster]().dict()
        
        if not isinstance(response, dict):
            return output
        
        for field, entities in response.items():
            if isinstance(entities, list) and field in output:
                output[field] = self._deduplicate_list(entities)
        
        return output

    def _deduplicate_list(self, entities: List[str]) -> List[str]:
        """Remove duplicates (case-insensitive) while preserving order."""
        seen = set()
        unique = []
        for e in entities:
            e_str = str(e).strip()
            if e_str and e_str.lower() not in seen:
                seen.add(e_str.lower())
                unique.append(e_str)
        return unique

    def _stage2_filter_consolidate(self, text: str, stage1_output: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
        """Stage 2: Filter & Consolidate entities.
        
        Apply hierarchical filtering (prioritize deeper semantic types).
        LLM-based validation is disabled as it causes hallucinations.
        """
        # Count Stage 1 entities
        stage1_count = sum(
            len(entities) for cluster in stage1_output.values() 
            for entities in cluster.values() if isinstance(entities, list)
        )
        print(f"\n[Stage 2] Input from Stage 1: {stage1_count} entities")
        
        # Apply hierarchical filtering
        hierarchy = get_hierarchy()
        hierarchical_filtered = {}
        for cluster_name, cluster_data in stage1_output.items():
            before_filter = sum(len(v) for v in cluster_data.values() if isinstance(v, list))
            hierarchical_filtered[cluster_name] = hierarchy.filter_by_depth(cluster_data)
            after_filter = sum(len(v) for v in hierarchical_filtered[cluster_name].values() if isinstance(v, list))
            
            if before_filter != after_filter:
                print(f"[Stage 2] {cluster_name}: {before_filter} -> {after_filter} entities (removed {before_filter - after_filter} duplicates)")
        
        # Count final entities
        final_count = sum(
            len(entities) for cluster in hierarchical_filtered.values() 
            for entities in cluster.values() if isinstance(entities, list)
        )
        print(f"[Stage 2] Output after hierarchical filter: {final_count} entities\n")
        # Additional textual evidence check: remove entities not present verbatim in the original text
        text_norm = (text or "").lower()
        evidence_filtered: Dict[str, Dict[str, List[str]]] = {}
        removed = 0
        for cluster_name, cluster_data in hierarchical_filtered.items():
            evidence_filtered[cluster_name] = {}
            for semantic_type, entities in cluster_data.items():
                kept = []
                for ent in entities:
                    try:
                        ent_norm = str(ent).strip().lower()
                    except Exception:
                        ent_norm = ''

                    # Simple heuristics to discard noisy extractions
                    too_long = len(ent_norm.split()) > 10 or len(ent_norm) > 120
                    has_colon = ':' in ent_norm or '\n' in ent_norm

                    if not ent_norm:
                        removed += 1
                        continue

                    if ent_norm in text_norm and (not too_long) and (not has_colon):
                        kept.append(ent)
                    else:
                        removed += 1
                evidence_filtered[cluster_name][semantic_type] = kept

        if removed:
            print(f"[Stage 2] Removed {removed} entities that lacked textual evidence")

        return evidence_filtered

    def _parse_filter_response(self, response: Any) -> List[Dict[str, str]]:
        """Parse filter response to extract validated entities."""
        if isinstance(response, dict) and "entities" in response:
            entities = response["entities"]
            if isinstance(entities, list):
                return [
                    {
                        "name": e.get("name", "") if isinstance(e, dict) else getattr(e, "name", ""),
                        "semantic_type": e.get("semantic_type", "") if isinstance(e, dict) else getattr(e, "semantic_type", ""),
                        "cluster": e.get("cluster", "") if isinstance(e, dict) else getattr(e, "cluster", "")
                    }
                    for e in entities
                ]
        return []

    def run_stage2_from_stage1(self, text: str, stage1_output: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
        """Public helper to run Stage 2 given a precomputed Stage 1 output.

        This avoids re-calling the LLM for Stage 1 and ensures Stage 2 operates
        deterministically on the same inputs used for Stage 1.
        """
        return self._stage2_filter_consolidate(text, stage1_output)

