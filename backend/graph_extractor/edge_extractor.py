import json
from typing import List, Dict, Optional, Union, Tuple
from functools import lru_cache
import sys
import os

# Add project root to sys.path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.schema import ExtractedEdges, ValidatedEntity, Entity, Edge
from backend.graph_extractor.prompts import EDGE_EXTRACTION_PROMPT, EDGE_VALIDATION_PROMPT
from backend.utils.time_logger import TimeLogger, Timer, setup_logger
from backend.krissbert_custom.usage.run_entity_linking import EntityLinker

logger = setup_logger("edge_extractor")

class EdgeExtractor:
    def __init__(self, llm_client, model_name: str, time_logger: Optional[TimeLogger] = None):
        self.llm_client = llm_client
        self.model_name = model_name
        self.time_logger = time_logger
        
        # Initialize Krissbert EntityLinker
        try:
            # Path to Krissbert model - assuming it's in the standard location or passed via config
            # For now hardcoding or using a default, ideally should be in config
            krissbert_path = "backend/krissbert_custom" 
            if not os.path.exists(os.path.join(krissbert_path, "pytorch_model.bin")):
                 # Fallback or check another path if needed, or just log warning
                 logger.warning(f"Krissbert model not found at {krissbert_path}, using 'bert-base-uncased' for testing/fallback")
                 krissbert_path = "bert-base-uncased"

            self.entity_linker = EntityLinker(
                model_name_or_path=krissbert_path,
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu" # Simple check, can be improved
            )
            # logger.info("EntityLinker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EntityLinker: {e}")
            self.entity_linker = None
    
    # Removed _cached_biosyn_predict as it is no longer used
    

    def extract(self, text: str, nodes: Union[List[Dict], ValidatedEntity], file_name: str = "unknown") -> Tuple[ExtractedEdges, List[Dict]]:  
        """Extract relationships using entities and compact UMLS features. Returns edges and new Level 2 nodes."""
        if not text or not text.strip():
            return ExtractedEdges(edges=[]), []

        if isinstance(nodes, list):
            entities_list = []
            for node in nodes:
                entities_list.append({
                    "name": node.get("name", ""),
                    "semantic_type": node.get("semantic_type", ""),
                    "mention": node.get("mention", ""),
                })
            nodes_dict = {"entities": entities_list}
        elif isinstance(nodes, ValidatedEntity):
            nodes_dict = nodes.dict()
            entities_list = nodes_dict.get("entities", [])
        else:
            return ExtractedEdges(edges=[]), []

        # Filter entities for LLM prompt (remove context, embeddings, etc.)
        prompt_entities = []
        for e in entities_list:
            prompt_entities.append({
                "name": e.get("name"),
                "semantic_type": e.get("semantic_type")
            })
        prompt_nodes_dict = {"entities": prompt_entities}

        # Krissbert Entity Linking & Level 2 Node Creation
        level2_nodes = []
        ref_to_edges = []
        
        if self.entity_linker:
            if self.time_logger:
                with Timer(self.time_logger, file_name, "Edge: Krissbert Linking"):
                    level2_nodes, ref_to_edges = self._process_krissbert_level2(entities_list)
            else:
                level2_nodes, ref_to_edges = self._process_krissbert_level2(entities_list)

        prompt = (
            EDGE_EXTRACTION_PROMPT
            .replace("[INPUT TEXT]", text)
            .replace("[ENTITIES LIST]", json.dumps(prompt_nodes_dict, ensure_ascii=False))
        )

        #save prompt to check 
        # with open ("edge_extraction_prompt.txt", "w", encoding="utf-8") as f:
        #     f.write(prompt) 
        
        def _generate_and_parse():
            resp = self.llm_client.generate(prompt=prompt)
            
            # Helper to convert LLM response to Edge objects
            def convert_to_edge(edge_data):
                if isinstance(edge_data, Edge):
                    return edge_data
                if not isinstance(edge_data, dict):
                    return None
                
                # Map subject/predicate/object (from prompt) to source/target/relation (schema)
                edge_dict = edge_data.copy()
                if "subject" in edge_dict:
                    edge_dict["source"] = edge_dict.pop("subject")
                if "predicate" in edge_dict:
                    edge_dict["relation"] = edge_dict.pop("predicate")
                if "object" in edge_dict:
                    edge_dict["target"] = edge_dict.pop("object")
                
                try:
                    return Edge(**edge_dict)
                except Exception:
                    return None
            
            # Normalize response to edges list
            edges_data = []
            
            if isinstance(resp, dict):
                edges_data = resp.get("edges", [])
                if not isinstance(edges_data, list):
                    edges_data = [edges_data] if edges_data else []
            
            elif isinstance(resp, list):
                # Raw list of edges
                edges_data = resp
            
            elif isinstance(resp, ExtractedEdges):
                return resp
            
            elif isinstance(resp, str):
                # Strip markdown code blocks
                cleaned_resp = resp.strip()
                if "```json" in cleaned_resp:
                    cleaned_resp = cleaned_resp.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned_resp:
                    cleaned_resp = cleaned_resp.split("```")[1].split("```")[0].strip()
                
                # logger.info(f"LLM Response (cleaned): {cleaned_resp}")
                
                try:
                    data = json.loads(cleaned_resp)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {cleaned_resp}")
                    data = {"edges": []}

                if isinstance(data, dict):
                    edges_data = data.get("edges", [])
                elif isinstance(data, list):
                    edges_data = data
                else:
                    edges_data = []
            
            # Convert all edges
            edge_objs = [convert_to_edge(e) for e in edges_data]
            edge_objs = [e for e in edge_objs if e is not None]
            return ExtractedEdges(edges=edge_objs)

        try:
            llm_edges_result = ExtractedEdges(edges=[])
            if self.time_logger:
                with Timer(self.time_logger, file_name, "Edge: LLM Generation"):
                    llm_edges_result = _generate_and_parse()
            else:
                llm_edges_result = _generate_and_parse()
            
            # Combine LLM edges with REF_TO edges
            all_edges = llm_edges_result.edges + ref_to_edges
            return ExtractedEdges(edges=all_edges), level2_nodes
            
        except Exception as e:
            import traceback
            logger.error(f"Edge extraction error: {e}\n{traceback.format_exc()}")
            # Still return REF_TO edges if any
            return ExtractedEdges(edges=ref_to_edges), level2_nodes

    def _process_krissbert_level2(self, entities: List[Dict]) -> Tuple[List[Dict], List[Edge]]:
        level2_nodes = []
        ref_to_edges = []
        
        # Prepare data for Krissbert
        # Krissbert expects: [{'mention': ..., 'context_left': ..., 'context_right': ...}]
        # We use Entity.name as 'mention' for linking, but we should probably use the actual mention text if available.
        # The user said: "mention trong krissbert là entity name ở bước node extraction" -> So use entity['name'] as mention.
        # But wait, context extraction relied on entity['mention'] (the span).
        # Let's use entity['name'] as the 'mention' field for Krissbert input as requested, 
        # but we need to pass the contexts we extracted.
        
        krissbert_input = []
        valid_entities = []
        
        for entity in entities:
            name = entity.get("name")
            if not name:
                continue
            
            krissbert_input.append({
                "mention": name, # User instruction: mention is entity name
                "context_left": entity.get("context_left", ""),
                "context_right": entity.get("context_right", "")
            })
            valid_entities.append(entity)
            
        if not krissbert_input:
            return [], []
            
        try:
            # Get top 5 candidates
            results = self.entity_linker.predict(krissbert_input, top_k=5)
            
            for i, res in enumerate(results):
                original_entity = valid_entities[i]
                candidates = res.get("candidates", [])
                
                # Filter by threshold
                candidates = [c for c in candidates if c.get('score', 0) >= 0.85]
                
                # We want to create Level 2 nodes from these candidates.
                # User example: {"cui":"C1436751","name":"...","definition":"...","icd":null,"semantic_types":[...]}
                # And create REF_TO edges.
                
                # Let's take the top 1 candidate for the primary REF_TO edge, 
                # or maybe create edges to all top 5? Usually linking is to the best match.
                # The user said "Tham khảo file... để biết cách query lấy top 5 node liên quan".
                # But for the graph, usually we link to the disambiguated entity.
                # Let's link to the top 1 for now to avoid explosion, or maybe top 3?
                # Let's stick to Top 1 for the "REF_TO" edge to keep the graph clean, 
                # but we can store others if needed. For now, Top 1.
                
                if candidates:
                    top_cand = candidates[0]
                    
                    # Create Level 2 Node
                    l2_node = {
                        "cui": top_cand.get("cui"),
                        "name": top_cand.get("name"),
                        "definition": top_cand.get("definition"),
                        "icd": top_cand.get("icd"),
                        "semantic_types": [], # Krissbert result might not have this, need to check payload
                        "level": "Level 2"
                    }
                    
                    # If semantic_types is in payload (it wasn't in the view_file of run_entity_linking, but maybe in Qdrant)
                    # The user example showed "semantic_types". 
                    # In run_entity_linking.py, payload fields were: cui, name, definition, icd.
                    # We might need to add semantic_types to run_entity_linking.py if it's in Qdrant payload.
                    # For now, let's leave it empty or try to get it if available.
                    if "semantic_types" in top_cand:
                         l2_node["semantic_types"] = top_cand["semantic_types"]
                    
                    level2_nodes.append(l2_node)
                    
                    # Create REF_TO edge
                    ref_edge = Edge(
                        source=original_entity.get("name"),
                        relation="REF_TO",
                        target=top_cand.get("name"),
                        evidence=f"Krissbert Entity Linking (Score: {top_cand.get('score', 0):.4f})"
                    )
                    ref_to_edges.append(ref_edge)

        except Exception as e:
            logger.error(f"Krissbert processing failed: {e}")
            
        return level2_nodes, ref_to_edges
