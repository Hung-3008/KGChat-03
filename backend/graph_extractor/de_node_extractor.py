import json
from typing import List, Dict, Optional
from backend.encoders.transformer_encoder import TransformerEncoder
from backend.graph_extractor.schema import (
    Activity, Phenomenon, PhysicalObject, ConceptualEntity, Entity, ValidatedEntity
)
from backend.graph_extractor.prompts import (
    ACTIVITY_PROMPT, PHENOMENON_PROMPT, PHYSICAL_OBJECT_PROMPT, CONCEPTUAL_ENTITY_PROMPT, CONTEXT_ENTITY_FILTER_PROMPT
)
from backend.graph_extractor.umls_hierarchy import (
    CLUSTER_DEFINITIONS, build_hierarchy_tree, filter_entities_by_hierarchy
)
from backend.utils.time_logger import TimeLogger, Timer, setup_logger

logger = setup_logger("node_extractor")

class NodeExtractor:
    def __init__(self, llm_client, model_name: str, embedding_model: str, encoder: Optional[TransformerEncoder] = None, device: str = "cpu", time_logger: Optional[TimeLogger] = None):
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.encoder = encoder or TransformerEncoder(model_name=embedding_model, device=device)
        self.hierarchy_tree = build_hierarchy_tree(CLUSTER_DEFINITIONS)
        self.time_logger = time_logger
    

    def get_mentioned_text(self, text: str, entities: List[Entity]) -> List[str]:
        pass

    def clean_empty_entities(self, entities: Dict) -> Dict:
        cleaned = {}
        for key, value in entities.items():
            if isinstance(value, list) and value:
                cleaned[key] = value
        return cleaned
    

    def extract_entities(self, text: str, prompt_template: str, output_schema) -> List[Dict]:
        if not text or not text.strip():
            return []
        
        prompt = prompt_template.replace("[INPUT TEXT]", text)
        try:
            resp = self.llm_client.generate(prompt=prompt, format=output_schema)
            return self.clean_empty_entities(resp)
        except Exception:
            return []
    

    def llm_filter_entities(self, text: str, entities: Dict) -> Dict:
        if not entities:
            return {}
        
        entities_json = json.dumps(entities)
        prompt = CONTEXT_ENTITY_FILTER_PROMPT.replace("[CLINICAL_INPUT_TEXT]", text).replace("[ENTITIES_INPUT]", entities_json)
        try:
            resp = self.llm_client.generate(prompt=prompt, format=ValidatedEntity)
            return self.clean_empty_entities(resp)
        except Exception:
            return {}

    def extract (self, text: str, file_name: str = "unknown") -> List[Dict]:
        """
        Stage 1: Extract raw medical entities from text using LLM, do per-cluster
        Stage 2: Hierarchically filter 
        Stage 3: LLM filter
        Stage 4: Embed entities
        """

        # Stage 1: Extract raw entities
        if self.time_logger:
            with Timer(self.time_logger, file_name, "Node Stage 1: Raw Extraction"):
                activity_entities = self.extract_entities(text, ACTIVITY_PROMPT, Activity)
                phenomenon_entities = self.extract_entities(text, PHENOMENON_PROMPT, Phenomenon)
                physical_object_entities = self.extract_entities(text, PHYSICAL_OBJECT_PROMPT, PhysicalObject)
                conceptual_entity_entities = self.extract_entities(text, CONCEPTUAL_ENTITY_PROMPT, ConceptualEntity)
        else:
            activity_entities = self.extract_entities(text, ACTIVITY_PROMPT, Activity)
            phenomenon_entities = self.extract_entities(text, PHENOMENON_PROMPT, Phenomenon)
            physical_object_entities = self.extract_entities(text, PHYSICAL_OBJECT_PROMPT, PhysicalObject)
            conceptual_entity_entities = self.extract_entities(text, CONCEPTUAL_ENTITY_PROMPT, ConceptualEntity)


        all_entities = {
            "activity": activity_entities,
            "phenomenon": phenomenon_entities,
            "physical_object": physical_object_entities,
            "conceptual_entity": conceptual_entity_entities,
        }

        #logger.info(f"Extracted Entities: {all_entities}")

        # Stage 2: Hierarchical filtering - filter entities by UMLS hierarchy depth
        if self.time_logger:
            with Timer(self.time_logger, file_name, "Node Stage 2: Hierarchy Filter"):
                filtered_entities = filter_entities_by_hierarchy(all_entities, self.hierarchy_tree)
        else:
            filtered_entities = filter_entities_by_hierarchy(all_entities, self.hierarchy_tree)
            
        #logger.info(f"Filtered Entities: {filtered_entities}")
        
        # Stage 3: LLM filtering
        if self.time_logger:
            with Timer(self.time_logger, file_name, "Node Stage 3: LLM Filter"):
                llm_filtered_entities = self.llm_filter_entities(text, filtered_entities)
        else:
            llm_filtered_entities = self.llm_filter_entities(text, filtered_entities)
            
        #logger.info(f"LLM Filtered Entities: {llm_filtered_entities}")
       
        # Stage 4: Embedding
        if self.time_logger:
            with Timer(self.time_logger, file_name, "Node Stage 4: Embedding"):
                entities_list = llm_filtered_entities.get("entities", []) if isinstance(llm_filtered_entities, dict) else []
                
                if not entities_list:
                    return []
                
                # Extract entity names
                names = []
                semantic_types = []
                for e in entities_list:
                    if isinstance(e, dict):
                        name = e.get("name", "").strip()
                        semantic_type = e.get("semantic_type", "").strip()
                    else:
                        name = getattr(e, "name", "").strip()
                        semantic_type = getattr(e, "semantic_type", "").strip()
                    if name:
                        names.append(name)
                        semantic_types.append(semantic_type)
                
                if not names:
                    return []
                
                # Generate embeddings for entity names
                try:
                    name_embeddings = self.encoder.embed_to_numpy(names).tolist()
                except Exception:
                    name_embeddings = []
        else:
            # Logic duplication avoided by better structure, but for now copy-paste with timer wrapper
            entities_list = llm_filtered_entities.get("entities", []) if isinstance(llm_filtered_entities, dict) else []
            if not entities_list:
                return []
            names = []
            semantic_types = []
            for e in entities_list:
                if isinstance(e, dict):
                    name = e.get("name", "").strip()
                    semantic_type = e.get("semantic_type", "").strip()
                else:
                    name = getattr(e, "name", "").strip()
                    semantic_type = getattr(e, "semantic_type", "").strip()
                if name:
                    names.append(name)
                    semantic_types.append(semantic_type)
            if not names:
                return []
            try:
                name_embeddings = self.encoder.embed_to_numpy(names).tolist()
            except Exception:
                name_embeddings = []
        
        # Combine entities with their embeddings
        output = []
        for i in range(len(names)):
            if i < len(name_embeddings):
                output.append({
                    "name": names[i],
                    "semantic_type": semantic_types[i],
                    "embedding": name_embeddings[i],
                })
        
        return output



        

