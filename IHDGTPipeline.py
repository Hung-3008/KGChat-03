import os
import sys
import yaml
import json
import logging
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils.neo4j_helper import Neo4jHelper
from backend.utils.qdrant_helper import QdrantHelper
from backend.utils.time_logger import setup_logger
from backend.encoders.transformer_encoder import TransformerEncoder
from backend.llm.llm_factory import LLMFactory

# Load environment variables
load_dotenv()

print("DEBUG: Script module loaded. Setting up logger...", flush=True)
logger = setup_logger("retrieval_pipeline", log_file=Path("debug_retrieval.log"))
print("DEBUG: Logger set up.", flush=True)

# ==========================================
# Schema Definitions
# ==========================================

class SubQuestion(BaseModel):
    question: str = Field(..., description="A self-contained sub-question derived from the original query.")
    reasoning: str = Field(..., description="Why this sub-question is important.")

class DecompositionResponse(BaseModel):
    sub_questions: List[SubQuestion] = Field(..., description="List of decomposed sub-questions.")

class Entity(BaseModel):
    name: str = Field(..., description="The medical entity name.")
    type: str = Field(..., description="The semantic type (e.g., Disease, Drug).")

class EntityLinkingResponse(BaseModel):
    entities: List[Entity] = Field(..., description="List of medical entities found in the text.")

class VerificationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the hypothesis is supported by the evidence.")
    reasoning: str = Field(..., description="Explanation of the verdict.")
    citation: Optional[str] = Field(None, description="Filename or source ID of the supporting evidence, if any.")


# ==========================================
# Pipeline Modules
# ==========================================

class ConfigManager:
    @staticmethod
    def load_config(path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

class QueryDecomposer:
    def __init__(self, llm_client, config: dict):
        self.llm = llm_client
        self.max_sub_questions = config.get("Decomposition", {}).get("max_sub_questions", 3)

    def decompose(self, query: str) -> List[SubQuestion]:
        logger.info(f"Decomposing query: {query}")
        prompt = f"""
        You are an expert medical AI. Decompose the following complex clinical query into atomic, self-contained sub-questions.
        Use a chain-of-thought approach. Limit to at most {self.max_sub_questions} sub-questions.
        
        Query: "{query}"
        """
        try:
            response = self.llm.generate(prompt, format=DecompositionResponse)
            if isinstance(response, dict):
                # Handle case where response is dict
                sub_qs = response.get("sub_questions", [])
                final_list = []
                for sq in sub_qs:
                    if isinstance(sq, SubQuestion):
                        final_list.append(sq)
                    elif isinstance(sq, dict):
                        final_list.append(SubQuestion(**sq))
                    else:
                        # Fallback or error?
                        continue
                return final_list
            # Handle case where response is object
            return response.sub_questions
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Fallback: treat original query as single sub-question
            return [SubQuestion(question=query, reasoning="Fallback due to decomposition error")]

class EntityLinker:
    def __init__(self, llm_client, config: dict):
        self.llm = llm_client
    
    def extract_entities(self, text: str) -> List[Entity]:
        prompt = f"""
        Extract key medical entities (Diseases, Symptoms, Drugs, Procedures, etc.) from the following text.
        Return them as a JSON list with 'name' and 'type'.
        
        Text: "{text}"
        """
        try:
            response = self.llm.generate(prompt, format=EntityLinkingResponse)
            if isinstance(response, dict):
                entities = response.get("entities", [])
                final_list = []
                for e in entities:
                    if isinstance(e, Entity):
                        final_list.append(e)
                    elif isinstance(e, dict):
                        final_list.append(Entity(**e))
                return final_list
            return response.entities
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")
            return []

class GraphTraverser:
    def __init__(self, neo4j_helper: Neo4jHelper, config: dict):
        self.neo4j = neo4j_helper
        self.relations = config.get("GraphTraversal", {}).get("relations", [])
        self.limit = config.get("GraphTraversal", {}).get("limit_per_hop", 10)

    def get_hypotheses(self, entities: List[Entity]) -> List[Dict]:
        hypotheses = []
        # Construct a WHERE clause for relations if specified
        rel_type_clause = ""
        if self.relations:
            rel_types = "|".join([f"`{r}`" for r in self.relations])
            rel_type_clause = f":{rel_types}"
        
        for entity in entities:
            # Fuzzy match or exact match query? 
            # We assume names are relatively clean or we use exact match for now.
            # Ideally, we should vector search for the node ID first, but for now specific Cypher.
            query = f"""
            MATCH (n:Level1 {{name: $name}})-[r{rel_type_clause}]-(m:Level1)
            RETURN n.name as head, type(r) as relation, m.name as tail, m.id as tail_id, r.source as source
            LIMIT {self.limit}
            """
            try:
                results = self.neo4j.query(query, {"name": entity.name})
                for r in results:
                    hypotheses.append({
                        "head": r['head'],
                        "relation": r['relation'],
                        "tail": r['tail'],
                        "tail_id": r['tail_id'],
                        "triple_str": f"{r['head']} {r['relation']} {r['tail']}"
                    })
            except Exception as e:
                logger.warning(f"Neo4j traversal failed for {entity.name}: {e}")
        
        return default_deduplicate(hypotheses)

def default_deduplicate(items: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for item in items:
        s = item.get("triple_str")
        if s and s not in seen:
            seen.add(s)
            unique.append(item)
    return unique

class EvidenceRetriever:
    def __init__(self, qdrant_helper: QdrantHelper, encoder: TransformerEncoder, config: dict):
        self.qdrant = qdrant_helper
        self.encoder = encoder
        self.collection_name = config.get("Retrieval", {}).get("collection_name", "kg_lv1_nodes")
        self.top_k = config.get("Retrieval", {}).get("top_k_dense", 5)

    def retrieve(self, query: str) -> List[Dict]:
        try:
            # Generate embedding
            # Encoder expects a list of strings
            vector_np = self.encoder.embed_to_numpy([query]) 
            vector = vector_np[0].tolist()
            
            # Use QdrantHelper's search wrapper
            # wrapper signature: search(collection_name, query_vector, limit, score_threshold)
            # It returns list of dicts: {"id": str, "name": str}
            # BUT we need payload for context!
            # The wrapper filters payload and only returns name? 
            # Let's check wrapper implementation again in thought...
            # Wrapper implementation:
            #             results = self.client.search(..., with_payload=True)
            #             filtered_results.append({"id": str(point.id), "name": payload.get("name", "Unknown")})
            # This is too limited for verification which needs context/definition.
            
            # Better to bypass wrapper for now and use client directly with correct method.
            # QdrantClient v1.x uses 'search'. v0.x might be different. 
            # If 'search' failed on 'client' object, maybe 'client' is not what we think?
            # In QdrantHelper __init__: self.client = QdrantClient(...)
            # Let's try `search` again but maybe the version is old?
            # Or maybe we should use `qdrant_helper.client.search` but ensure it's the sync client.
            # The error 'QdrantClient' object has no attribute 'search' is very specific.
            # It suggests the method is named something else, e.g. `search_batch` or `query`?
            # OR maybe it's `async` client?
            
            # Let's try using the helper's `search` method but we need to Modify helper to return payload?
            # Or just use the helper as is for now to fix the crash, acknowledging we might lose some context detail.
            # Wait, I should probably Fix QdrantHelper to return full payload or creating a new method there.
            # But I cannot edit qdrant_helper as it might break other things (though likely safe).
            
            # Let's call the helper's search method for safety, as we know it exists.
            # But we need to fetch logic.
            
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=self.top_k,
                score_threshold=0.0 # Get all top k
            )
            
            # The helper returns [{"id":..., "name":...}].
            # We need to reconstruct context. Verification prompts use "definition", "semantic_type".
            # The helper drops these. This is bad for verification.
            
            # Option B: Use the `self.qdrant.client` but use `search_groups` or `recommend`?
            # No, `search` is standard.
            # Maybe the version of qdrant-client installed is very old (0.11?) or very new?
            # Let's try `query_points`? `retrieve`?
            
            # Let's check `backend/utils/qdrant_helper.py` again. 
            # Line 103: results = self.client.search(...) 
            # So the wrapper DOES call `.search()` on `self.client`!
            # If `self.client.search` works inside `QdrantHelper.search`, why did it fail in `retrieval.py`?
            # Ah! In retrieval.py I called `self.qdrant.client.search`.
            # If QdrantHelper.search calls self.client.search and it works... 
            # Wait, did the logs show `QdrantHelper.search` working? No, I didn't call it.
            # But if the file exists and has that code, it should work.
            
            # HYPOTHESIS: `QdrantHelper.client` is NOT a `QdrantClient` instance directly?
            # No, line 18: `self.client = QdrantClient(...)`
            
            # HYPOTHESIS: `qdrant_client` version on this machine has `search` method?
            # If `qdrant_helper.py` has it, it probably exists.
            
            # Let's just use `self.qdrant.search(...)` since that method IS defined in the helper class.
            # And I will trust it works.
            # But I need to handle the limited return format.
            # Ideally I should update `QdrantHelper` to return full payload.
            
            # Let's use `self.qdrant.client.search` again but double check import.
            # Maybe I need to look at `dir(self.qdrant.client)` in a debug script.
            
            # For now, to unblock, I will use `self.qdrant.search` which is safe.
            # And adapt the result format.
            
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=self.top_k,
                score_threshold=0.65 # Use threshold from helper default or config
            )
            
            # Helper now returns {"id", "name", "payload", "score"}
            # We can use it directly or map if needed.
            # Start loop to ensure format match just in case
            results = []
            for item in search_result:
                results.append({
                    "score": item["score"],
                    "payload": item["payload"],
                    "id": item["id"]
                })
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

class Verifier:
    def __init__(self, llm_client, config: dict):
        self.llm = llm_client

    def verify(self, hypothesis: Dict, evidence: List[Dict]) -> Dict:
        # Flatten evidence
        context = "\n".join([f"- {e['payload'].get('definition', '') or e['payload'].get('name', '')} (Type: {e['payload'].get('semantic_type', '')})" for e in evidence])
        
        prompt = f"""
        Act as a medical judge. Verify if the following graph triple is clinically valid based on the provided evidence context.
        
        Triple: {hypothesis['triple_str']}
        
        Evidence Context:
        {context}
        
        Return JSON with 'is_valid' (bool), 'reasoning' (str), and 'citation' (str, optional source from context).
        If the evidence doesn't explicitly support it but it is generally known medical fact, you may lean towards true but note it in reasoning.
        """
        try:
            response = self.llm.generate(prompt, format=VerificationResponse)
            res_dict = response if isinstance(response, dict) else response.model_dump()
            return res_dict
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"is_valid": False, "reasoning": "Verification Error"}

class Responser:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_answer(self, query: str, context: List[Dict]) -> str:
        # Format verified paths
        facts = []
        for item in context:
            facts.append(f"Fact: {item['path']['triple_str']}\n- Reasoning: {item['verification']['reasoning']}")
        
        facts_str = "\n".join(facts)
        
        prompt = f"""
        Answer the user query based on the verified facts below. Cite the facts where appropriate.
        
        Query: {query}
        
        Verified Facts:
        {facts_str}
        
        Answer:
        """
        return self.llm.generate(prompt)

# ==========================================
# Main Orchestrator
# ==========================================

class IHDGTPipeline:
    def __init__(self, config_path: str):
        logger.info(f"Loading config from {config_path}...")
        self.config = ConfigManager.load_config(config_path)
        
        # Init components
        logger.info("Initializing LLM Client...")
        llm_config = self.config.get("Retrieval", {}).get("LLM", {})
        self.llm = LLMFactory.create_client(llm_config)
        
        # Helpers
        logger.info("Connecting to Neo4j...")
        self.neo4j = Neo4jHelper()
        logger.info("Connecting to Qdrant...")
        self.qdrant = QdrantHelper()
        
        logger.info("Loading Transformer Encoder (this may take a moment)...")
        self.encoder = TransformerEncoder(
            model_name=self.config.get("Encoder", {}).get("model_name"),
            device=self.config.get("Encoder", {}).get("device")
        )
        self.encoder._ensure_model_loaded()
        logger.info("Encoder loaded.")

        # Modules
        self.decomposer = QueryDecomposer(self.llm, self.config)
        self.linker = EntityLinker(self.llm, self.config)
        self.traverser = GraphTraverser(self.neo4j, self.config)
        self.retriever = EvidenceRetriever(self.qdrant, self.encoder, self.config)
        self.verifier = Verifier(self.llm, self.config)
        self.responser = Responser(self.llm)
        
        # Concurrency
        self.concurrency = llm_config.get("concurrency", 5)
        logger.info(f"Pipeline initialized with concurrency={self.concurrency}")

    def run(self, query: str):
        logger.info(">>> Step 1: Decomposition")
        sub_questions = self.decomposer.decompose(query)
        logger.info(f"Generated {len(sub_questions)} sub-questions.")
        
        final_verified_paths = []
        
        def process_sub_question(sq: SubQuestion):
            try:
                logger.info(f"[Start] Processing SubQ: {sq.question}")
                
                # Step 2: Entity Linking
                entities = self.linker.extract_entities(sq.question)
                logger.info(f"[Link] Found {len(entities)} entities for '{sq.question}'")
                
                # Step 3: Hypothesis Generation (Graph Lookup)
                hypotheses = self.traverser.get_hypotheses(entities)
                logger.info(f"[Graph] Generated {len(hypotheses)} hypotheses for '{sq.question}'")
                
                sub_results = []
                
                # Step 4 & 5: Retrieval & Verification
                for i, hyp in enumerate(hypotheses):
                    verification_query = f"Does {hyp['triple_str']} in clinical context?"
                    # logger.info(f"[Retrieve] Checking hypothesis {i+1}/{len(hypotheses)}: {hyp['triple_str']}")
                    evidence = self.retriever.retrieve(verification_query)
                    
                    # Check validity
                    verdict = self.verifier.verify(hyp, evidence)
                    
                    if verdict.get("is_valid", False):
                        logger.info(f"  [Verify] VALID: {hyp['triple_str']}")
                        sub_results.append({
                            "path": hyp,
                            "evidence": evidence,
                            "verification": verdict
                        })
                    else:
                         # logger.info(f"  [Verify] INVALID: {hyp['triple_str']}")
                         pass
                        
                logger.info(f"[End] Finished SubQ: {sq.question} - Found {len(sub_results)} verified paths")
                return sub_results
            except Exception as e:
                logger.error(f"Error in process_sub_question '{sq.question}': {e}")
                return []

        # Use ThreadPool for sub-questions if > 1
        # NOTE: Reduced concurrency for debugging if needed, but keeping dynamic
        logger.info(f"Starting parallel processing of {len(sub_questions)} sub-questions...")
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [executor.submit(process_sub_question, sq) for sq in sub_questions]
            for future in futures:
                try:
                    res = future.result()
                    final_verified_paths.extend(res)
                except Exception as e:
                    logger.error(f"Error processing sub-question future: {e}")

        logger.info(f"Total Verified Paths: {len(final_verified_paths)}")
        
        # Step 6: Synthesis
        logger.info(">>> Step 6: Synthesis")
        if not final_verified_paths:
             logger.warning("No verified paths found. potentially answering with fallback.")
             return "I could not find enough verified information to answer your query securely."
             
        answer = self.responser.generate_answer(query, final_verified_paths)
        return answer

    def cleanup(self):
        self.neo4j.close()

def main():
    parser = argparse.ArgumentParser(description="IHD-GT Retrieval Pipeline")
    parser.add_argument("--query", type=str, required=True, help="Clinical query")
    parser.add_argument("--config", type=str, default="backend/configs/ihdpgt_config.yml", help="Path to config")
    
    args = parser.parse_args()
    
    pipeline = None
    try:
        pipeline = IHDGTPipeline(args.config)
        answer = pipeline.run(args.query)
        print("\n=== FINAL ANSWER ===\n")
        print(answer)
        print("\n====================\n")
    except Exception as e:
        logger.error(f"Pipeline Execution Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline.cleanup()

if __name__ == "__main__":
    main()
