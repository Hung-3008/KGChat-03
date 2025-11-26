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
from backend.utils.umls_entity_lookup import build_umls_prompt_features, UMLSEntityLookup
from backend.utils.time_logger import TimeLogger, Timer, setup_logger
from backend.BioSyn.inference import BioSynInference

logger = setup_logger("edge_extractor")

class EdgeExtractor:
    def __init__(self, llm_client, model_name: str, time_logger: Optional[TimeLogger] = None):
        self.llm_client = llm_client
        self.model_name = model_name
        self.time_logger = time_logger
        
        # Initialize BioSyn
        try:
            self.biosyn = BioSynInference()
            # logger.info("BioSynInference initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize BioSynInference: {e}")
            self.biosyn = None
    
    @lru_cache(maxsize=1000)
    def _cached_biosyn_predict(self, mention: str) -> str:
        """Cached BioSyn prediction to avoid redundant computations"""
        if not self.biosyn:
            return "{}"
        try:
            import json
            result = self.biosyn.predict(mention)
            return json.dumps(result)
        except Exception as e:
            logger.warning(f"BioSyn prediction failed for '{mention}': {e}")
            return "{}"
    

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

        # Build compact UMLS features for the current entities
        if self.time_logger:
            with Timer(self.time_logger, file_name, "Edge: UMLS Lookup"):
                umls_features = build_umls_prompt_features(
                    entities_list,
                    db_path="data/umls.duckdb",
                    max_candidates_per_entity=1,
                    max_relations_per_pair=2,
                )
        else:
            umls_features = build_umls_prompt_features(
                entities_list,
                db_path="data/umls.duckdb",
                max_candidates_per_entity=1,
                max_relations_per_pair=2,
            )

        # BioSyn Entity Linking & Level 2 Node Creation
        biosyn_hints = []
        level2_nodes = []
        ref_to_edges = []
        
        if self.biosyn:
            if self.time_logger:
                with Timer(self.time_logger, file_name, "Edge: BioSyn & Level 2"):
                    biosyn_hints, level2_nodes, ref_to_edges = self._process_biosyn_level2(entities_list)
            else:
                biosyn_hints, level2_nodes, ref_to_edges = self._process_biosyn_level2(entities_list)

        prompt = (
            EDGE_EXTRACTION_PROMPT
            .replace("[INPUT TEXT]", text)
            .replace("[ENTITIES LIST]", json.dumps(nodes_dict, ensure_ascii=False))
            .replace("[UMLS NODES]", json.dumps(umls_features.get("nodes", []), ensure_ascii=False))
            .replace("[UMLS EDGES]", json.dumps(umls_features.get("edges", []), ensure_ascii=False))
            .replace("[BIOSYN HINTS]", json.dumps(biosyn_hints, ensure_ascii=False))
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

    def _process_biosyn_level2(self, entities: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Edge]]:
        hints = []
        level2_nodes = []
        ref_to_edges = []
        
        # Step 1: Collect all BioSyn predictions and CUIs
        all_predictions = []  # List of (entity, predictions)
        
        for entity in entities:
            mention = entity.get("mention") or entity.get("name")
            if not mention:
                continue
            
            try:
                result_json = self._cached_biosyn_predict(mention)
                result = json.loads(result_json)
                predictions = result.get("predictions", [])
                
                if not predictions:
                    continue
                    
                # Hints for LLM (only top 1)
                top_pred = predictions[0]
                hints.append({
                    "entity": mention,
                    "biosyn_cui": top_pred.get("id"),
                    "biosyn_name": top_pred.get("name")
                })
                
                # Store entity and its top 3 predictions
                all_predictions.append((entity, predictions[:3]))
                    
            except Exception as e:
                logger.warning(f"BioSyn processing failed for '{mention}': {e}")
        
        # Step 2: Batch query definitions and ICD codes
        def clean_id(mid):
            if mid and "|" in mid:
                return mid.split("|")[-1]
            return mid

        all_mesh_ids = [clean_id(pred.get("id")) for _, preds in all_predictions for pred in preds]
        
        icd_map = {}
        def_map = {}
        mesh_to_cui_map = {}
        
        if all_mesh_ids:
            with UMLSEntityLookup("data/umls.duckdb") as lookup:
                # Map MeSH IDs to CUIs
                mesh_to_cui_map = lookup.map_mesh_ids_to_cuis(all_mesh_ids)
                
                # Get unique CUIs
                all_cuis = list(set(mesh_to_cui_map.values()))
                
                if all_cuis:
                    icd_map = lookup.get_icd_codes_batch(all_cuis)
                    def_map = lookup.get_definitions_batch(all_cuis)
        
        # Step 3: Create Level 2 nodes and REF_TO edges using batch results
        for entity, predictions in all_predictions:
            for pred in predictions:
                mesh_id = clean_id(pred.get("id"))
                name = pred.get("name")
                
                # Get mapped CUI
                cui = mesh_to_cui_map.get(mesh_id)
                
                # Lookup from batch results using CUI
                definition = def_map.get(cui, "") if cui else ""
                icd = icd_map.get(cui, "") if cui else ""
                
                # Create Level 2 Node
                l2_node = {
                    "name": name,
                    "semantic_type": "Level 2",
                    "definition": definition,
                    "icd": icd,
                    "cui": cui if cui else mesh_id, # Prefer UMLS CUI, fallback to MeSH ID
                    "level": "Level 2"
                }
                level2_nodes.append(l2_node)
                
                # Create REF_TO edge
                ref_edge = Edge(
                    source=entity.get("name"),
                    relation="REF_TO",
                    target=name,
                    evidence="BioSyn Entity Linking"
                )
                ref_to_edges.append(ref_edge)
                
        return hints, level2_nodes, ref_to_edges
