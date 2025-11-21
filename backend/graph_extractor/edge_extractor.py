import json
from typing import List, Dict, Optional, Union, Tuple
from backend.graph_extractor.schema import ExtractedEdges, ValidatedEntity, Entity, Edge
from backend.graph_extractor.prompts import EDGE_EXTRACTION_PROMPT, EDGE_VALIDATION_PROMPT
from backend.utils.umls_entity_lookup import build_umls_prompt_features 
from backend.utils.time_logger import TimeLogger, Timer, setup_logger

logger = setup_logger("edge_extractor")

class EdgeExtractor:
    def __init__(self, llm_client, model_name: str, time_logger: Optional[TimeLogger] = None):
        self.llm_client = llm_client
        self.model_name = model_name
        self.time_logger = time_logger
    

    def extract(self, text: str, nodes: Union[List[Dict], ValidatedEntity], file_name: str = "unknown") -> ExtractedEdges:  
        """Extract relationships using entities and compact UMLS features."""
        if not text or not text.strip():
            return ExtractedEdges(edges=[])

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
            return ExtractedEdges(edges=[])

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

        prompt = (
            EDGE_EXTRACTION_PROMPT
            .replace("[INPUT TEXT]", text)
            .replace("[ENTITIES LIST]", json.dumps(nodes_dict, ensure_ascii=False))
            .replace("[UMLS NODES]", json.dumps(umls_features.get("nodes", []), ensure_ascii=False))
            .replace("[UMLS EDGES]", json.dumps(umls_features.get("edges", []), ensure_ascii=False))
        )

        #save prompt to check 
        # with open ("edge_extraction_prompt.txt", "w", encoding="utf-8") as f:
        #     f.write(prompt) 
        
        def _generate_and_parse():
            resp = self.llm_client.generate(prompt=prompt, format=ExtractedEdges)
            
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
                data = json.loads(resp)
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
            if self.time_logger:
                with Timer(self.time_logger, file_name, "Edge: LLM Generation"):
                    return _generate_and_parse()
            else:
                return _generate_and_parse()
            
        except Exception as e:
            logger.error(f"Edge extraction error: {e}")
            return ExtractedEdges(edges=[])
