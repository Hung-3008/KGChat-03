import json
from typing import List, Dict, Optional, Union, Tuple
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
    

    def extract_context(self, text: str, mention: str, window_size: int = 50) -> Tuple[str, str]:
        if not mention or mention not in text:
            return "", ""
        
        start_idx = text.find(mention)
        end_idx = start_idx + len(mention)
        
        # Extract left context
        left_start = max(0, start_idx - window_size)
        context_left = text[left_start:start_idx]
        
        # Extract right context
        right_end = min(len(text), end_idx + window_size)
        context_right = text[end_idx:right_end]
        
        return context_left, context_right

    def clean_empty_entities(self, entities: Union[Dict, List]) -> Dict:
        if isinstance(entities, list):
            if entities and isinstance(entities[0], dict):
                entities = entities[0]
            else:
                return {}
                
        cleaned = {}
        if not isinstance(entities, dict):
             return {}
             
        for key, value in entities.items():
            if isinstance(value, list) and value:
                cleaned_values = []
                for item in value:
                    if isinstance(item, dict):
                        # If it has semantic_type, keep the dict (Stage 3 output)
                        if "semantic_type" in item and ("name" in item or "mention" in item):
                            cleaned_values.append(item)
                        # Otherwise extract name (Stage 1 output might be wrapped)
                        elif "name" in item:
                            cleaned_values.append(item["name"])
                        elif "mention" in item:
                            cleaned_values.append(item["mention"])
                        elif item:
                            cleaned_values.append(str(list(item.values())[0]))
                    elif isinstance(item, str):
                        cleaned_values.append(item)
                
                if cleaned_values:
                    cleaned[key] = cleaned_values
        return cleaned
    

    def build_simple_schema(self, model_class) -> Dict:
        """Manually build JSON schema to avoid Pydantic crash."""
        properties = {}
        for name, field in model_class.model_fields.items():
            # All fields in cluster models are List[str]
            properties[name] = {"type": "array", "items": {"type": "string"}}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys())
        }

    def extract_entities(self, text: str, prompt_template: str, output_schema) -> List[Dict]:
        if not text or not text.strip():
            return []
        
        prompt = prompt_template.replace("[INPUT TEXT]", text)
        try:
            # Build schema manually
            schema = self.build_simple_schema(output_schema)
            resp = self.llm_client.generate(prompt=prompt, format=schema)
            # logger.info(f"NodeExtractor LLM Resp: {resp}")
            
            # Resp should be a dict now since we passed format
            if isinstance(resp, dict):
                 return self.clean_empty_entities(resp)
            elif isinstance(resp, str):
                 # Fallback if client didn't parse it
                 try:
                     return self.clean_empty_entities(json.loads(resp))
                 except:
                     return {}
            else:
                 return {}

        except Exception as e:
            logger.error(f"Error in extract_entities: {e}")
            return {}
    

    def llm_filter_entities(self, text: str, entities: Dict) -> Dict:
        if not entities:
            return {}
        
        entities_json = json.dumps(entities)
        prompt = CONTEXT_ENTITY_FILTER_PROMPT.replace("[CLINICAL_INPUT_TEXT]", text).replace("[ENTITIES_INPUT]", entities_json)
        
        # Manual schema for ValidatedEntity
        schema = {
          "type": "object",
          "properties": {
            "entities": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "semantic_type": {"type": "string"},
                  "mention": {"type": "string"}
                },
                "required": ["name", "semantic_type", "mention"]
              }
            }
          },
          "required": ["entities"]
        }

        try:
            resp = self.llm_client.generate(prompt=prompt, format=schema)
            # logger.info(f"LLM Filter Resp: {resp}")
            
            if isinstance(resp, dict):
                 return self.clean_empty_entities(resp)
            elif isinstance(resp, str):
                 try:
                     return self.clean_empty_entities(json.loads(resp))
                 except:
                     return {}
            else:
                 return {}

        except Exception as e:
            logger.error(f"Error in llm_filter_entities: {e}")
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
            with Timer(self.time_logger, file_name, "node_stage1"):
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
            with Timer(self.time_logger, file_name, "node_stage2"):
                filtered_entities = filter_entities_by_hierarchy(all_entities, self.hierarchy_tree)
        else:
            filtered_entities = filter_entities_by_hierarchy(all_entities, self.hierarchy_tree)
            
        #logger.info(f"Filtered Entities: {filtered_entities}")
        
        # Stage 3: LLM filtering
        if self.time_logger:
            with Timer(self.time_logger, file_name, "node_stage3"):
                llm_filtered_entities = self.llm_filter_entities(text, filtered_entities)
        else:
            llm_filtered_entities = self.llm_filter_entities(text, filtered_entities)
            
        #logger.info(f"LLM Filtered Entities: {llm_filtered_entities}")
       
        # Stage 4: Embedding
        if self.time_logger:
            with Timer(self.time_logger, file_name, "node_stage4"):
                entities_list = []
                if isinstance(llm_filtered_entities, dict):
                    if "entities" in llm_filtered_entities:
                        entities_list = llm_filtered_entities["entities"]
                    else:
                        # Flatten grouped structure
                        for cluster, types in llm_filtered_entities.items():
                            if isinstance(types, dict):
                                for sem_type, names in types.items():
                                    if isinstance(names, list):
                                        for name in names:
                                            if isinstance(name, str):
                                                entities_list.append({"name": name, "semantic_type": sem_type})
                                            elif isinstance(name, dict):
                                                 n = name.get("name") or name.get("mention")
                                                 if n:
                                                     entities_list.append({"name": n, "semantic_type": sem_type})
                
                if not entities_list:
                    return []
                
                # Extract entity names
                names = []
                semantic_types = []
                mentions = []
                contexts_left = []
                contexts_right = []
                
                for e in entities_list:
                    if isinstance(e, dict):
                        name = e.get("name", "").strip()
                        semantic_type = e.get("semantic_type", "").strip()
                        mention = e.get("mention", "").strip()
                    else:
                        name = getattr(e, "name", "").strip()
                        semantic_type = getattr(e, "semantic_type", "").strip()
                        mention = getattr(e, "mention", "").strip()
                    if name:
                        names.append(name)
                        semantic_types.append(semantic_type)
                        
                        # Extract context
                        if not mention:
                            mention = name # Fallback
                        mentions.append(mention)
                        
                        ctx_left, ctx_right = self.extract_context(text, mention)
                        contexts_left.append(ctx_left)
                        contexts_right.append(ctx_right)
                
                if not names:
                    return []
                
                # Generate embeddings for entity names
                try:
                    name_embeddings = self.encoder.embed_to_numpy(names).tolist()
                except Exception:
                    name_embeddings = []
        else:
            # Logic duplication avoided by better structure, but for now copy-paste with timer wrapper
            entities_list = []
            if isinstance(llm_filtered_entities, dict):
                if "entities" in llm_filtered_entities:
                    entities_list = llm_filtered_entities["entities"]
                else:
                    # Flatten grouped structure
                    for cluster, types in llm_filtered_entities.items():
                        if isinstance(types, dict):
                            for sem_type, names in types.items():
                                if isinstance(names, list):
                                    for name in names:
                                        if isinstance(name, str):
                                            entities_list.append({"name": name, "semantic_type": sem_type})
                                        elif isinstance(name, dict):
                                             # Handle if name is dict
                                             n = name.get("name") or name.get("mention")
                                             if n:
                                                 entities_list.append({"name": n, "semantic_type": sem_type})

            if not entities_list:
                return []
            names = []
            semantic_types = []
            mentions = []
            contexts_left = []
            contexts_right = []
            
            for e in entities_list:
                if isinstance(e, dict):
                    name = e.get("name", "").strip()
                    semantic_type = e.get("semantic_type", "").strip()
                    mention = e.get("mention", "").strip()
                else:
                    name = getattr(e, "name", "").strip()
                    semantic_type = getattr(e, "semantic_type", "").strip()
                    mention = getattr(e, "mention", "").strip()
                
                if name:
                    names.append(name)
                    semantic_types.append(semantic_type)
                    
                    # Extract context
                    if not mention:
                        mention = name # Fallback
                    mentions.append(mention)
                    
                    ctx_left, ctx_right = self.extract_context(text, mention)
                    contexts_left.append(ctx_left)
                    contexts_right.append(ctx_right)

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
                    "mention": mentions[i] if i < len(mentions) else names[i],
                    "context_left": contexts_left[i] if i < len(contexts_left) else "",
                    "context_right": contexts_right[i] if i < len(contexts_right) else "",
                })
        
        return output



        

