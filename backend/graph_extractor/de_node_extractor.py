import json
from typing import List, Dict, Optional, Union, Tuple
from backend.encoders.transformer_encoder import TransformerEncoder
from backend.graph_extractor.schema import (
    Activity, Phenomenon, PhysicalObject, ConceptualEntity, Entity, ValidatedEntity
)
from backend.graph_extractor.prompts import (
    CONTEXT_ENTITY_FILTER_PROMPT, COMBINED_ENTITY_PROMPT
)
from backend.graph_extractor.umls_hierarchy import (
    CLUSTER_DEFINITIONS, build_hierarchy_tree, filter_entities_by_hierarchy
)
from backend.utils.time_logger import TimeLogger, Timer, setup_logger

logger = setup_logger("node_extractor")

class NodeExtractor:
    def __init__(self, llm_client, model_name: str, embedding_model: str, encoder: Optional[TransformerEncoder] = None, device: str = "cpu", embed_batch_size: int = 64):
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.encoder = encoder or TransformerEncoder(model_name=embedding_model, device=device)
        self.hierarchy_tree = build_hierarchy_tree(CLUSTER_DEFINITIONS)
        # self.time_logger removed from init to encourage stateless passing
        self.combined_schema = self.build_combined_schema()
        self.embed_batch_size = embed_batch_size
    

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
        """Backward-compatible cleaner for flat dict schemas (kept for Stage 3)."""
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
                        if "semantic_type" in item and ("name" in item or "mention" in item):
                            cleaned_values.append(item)
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

    def build_combined_schema(self) -> Dict:
        """Construct a strict JSON schema merging all clusters for one-shot extraction."""
        clusters = {
            "activity": Activity,
            "phenomenon": Phenomenon,
            "physical_object": PhysicalObject,
            "conceptual_entity": ConceptualEntity,
        }

        properties: Dict[str, Dict] = {}
        for cluster_name, model_cls in clusters.items():
            cluster_props: Dict[str, Dict] = {}
            for field_name in model_cls.model_fields.keys():
                cluster_props[field_name] = {"type": "array", "items": {"type": "string"}}
            properties[cluster_name] = {
                "type": "object",
                "properties": cluster_props,
                "required": list(cluster_props.keys())
            }

        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys())
        }

    def _clean_combined_entities(self, resp: Dict) -> Dict[str, Dict[str, List[str]]]:
        if not isinstance(resp, dict):
            return {}

        cleaned: Dict[str, Dict[str, List[str]]] = {}
        for cluster, cluster_data in resp.items():
            if not isinstance(cluster_data, dict):
                continue
            cluster_cleaned: Dict[str, List[str]] = {}
            for sem_type, values in cluster_data.items():
                if isinstance(values, list):
                    normalized = []
                    for v in values:
                        if isinstance(v, str):
                            v_clean = v.strip()
                            if v_clean:
                                normalized.append(v_clean)
                        elif isinstance(v, (int, float)):
                            normalized.append(str(v))
                    if normalized:
                        cluster_cleaned[sem_type] = normalized
            if cluster_cleaned:
                cleaned[cluster] = cluster_cleaned
        return cleaned
    

    def extract_all_entities(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        if not text or not text.strip():
            return {}

        prompt = COMBINED_ENTITY_PROMPT.replace("[INPUT TEXT]", text)
        try:
            resp = self.llm_client.generate(prompt=prompt, format=self.combined_schema)
            if isinstance(resp, dict):
                return self._clean_combined_entities(resp)
            if isinstance(resp, str):
                try:
                    return self._clean_combined_entities(json.loads(resp))
                except Exception:
                    return {}
            return {}
        except Exception as e:
            logger.error(f"Error in extract_all_entities: {e}")
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

    def extract (self, text: str, file_name: str = "unknown", time_logger: Optional[TimeLogger] = None) -> List[Dict]:
        """
        Stage 1: Extract raw medical entities from text using LLM, do per-cluster
        Stage 2: Hierarchically filter 
        Stage 3: LLM filter
        Stage 4: Embed entities
        """

        # Stage 1: Extract raw entities in a single structured call
        if time_logger:
            with Timer(time_logger, file_name, "node_stage1"):
                all_entities = self.extract_all_entities(text)
        else:
            all_entities = self.extract_all_entities(text)

        #logger.info(f"Extracted Entities: {all_entities}")

        # Stage 2: Hierarchical filtering - filter entities by UMLS hierarchy depth
        if time_logger:
            with Timer(time_logger, file_name, "node_stage2"):
                filtered_entities = filter_entities_by_hierarchy(all_entities, self.hierarchy_tree)
        else:
            filtered_entities = filter_entities_by_hierarchy(all_entities, self.hierarchy_tree)
            
        #logger.info(f"Filtered Entities: {filtered_entities}")
        
        # Stage 3: LLM filtering
        if time_logger:
            with Timer(time_logger, file_name, "node_stage3"):
                llm_filtered_entities = self.llm_filter_entities(text, filtered_entities)
        else:
            llm_filtered_entities = self.llm_filter_entities(text, filtered_entities)
            
        #logger.info(f"LLM Filtered Entities: {llm_filtered_entities}")
       
        # Stage 4: Embedding
        if time_logger:
            with Timer(time_logger, file_name, "node_stage4"):
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
                    name_embeddings = self.encoder.embed_to_numpy(names, batch_size=self.embed_batch_size).tolist()
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
                name_embeddings = self.encoder.embed_to_numpy(names, batch_size=self.embed_batch_size).tolist()
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



        

