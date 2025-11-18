import json
from typing import List, Dict, Optional, Union
from backend.graph_extractor.schema import ExtractedEdges, ValidatedEntity, Entity
from backend.graph_extractor.prompts import EDGE_EXTRACTION_PROMPT


class EdgeExtractor:
    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
    

    def extract(self, text: str, nodes: Union[List[Dict], ValidatedEntity]) -> ExtractedEdges:  
        if not text or not text.strip():
            return ExtractedEdges(edges=[])
        
        
        if isinstance(nodes, list):
            
            entities_list = []
            for node in nodes:
                entities_list.append({
                    "name": node.get("name", ""),
                    "semantic_type": node.get("semantic_type", ""),
                    "mention": ""  
                })
            nodes_dict = {"entities": entities_list}
        elif isinstance(nodes, ValidatedEntity):
            nodes_dict = nodes.dict()
        else:
            return ExtractedEdges(edges=[])
        
        prompt = EDGE_EXTRACTION_PROMPT.replace("[INPUT TEXT]", text).replace("[ENTITIES LIST]", json.dumps(nodes_dict, ensure_ascii=False))  
        try:
            resp = self.llm_client.generate(prompt=prompt, format=ExtractedEdges)
            return resp
        except Exception as e:
            print(f"Edge extraction error: {e}")
            return ExtractedEdges(edges=[])