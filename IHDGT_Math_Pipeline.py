
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

# New Math Plugin
from backend.math.ppr_solver import PPRSolver

# Reuse existing schema/classes from IHDGTPipeline where possible?
# Ideally, we should import them to avoid duplication, but IHDGTPipeline is a script, not a module structure.
# So I will redefine or copy them here for self-containment as requested.

# Load environment variables
load_dotenv()

print("DEBUG: Math Pipeline loaded. Setting up logger...", flush=True)
logger = setup_logger("math_pipeline", log_file=Path("debug_math_pipeline.log"))
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
# Pipeline Modules (Standard)
# ==========================================

class ConfigManager:
    @staticmethod
    def load_config(path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

# Reusing Logic for common parts
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
                sub_qs = response.get("sub_questions", [])
                final_list = []
                for sq in sub_qs:
                    if isinstance(sq, SubQuestion): final_list.append(sq)
                    elif isinstance(sq, dict): final_list.append(SubQuestion(**sq))
                return final_list
            return response.sub_questions
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
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
                    if isinstance(e, Entity): final_list.append(e)
                    elif isinstance(e, dict): final_list.append(Entity(**e))
                return final_list
            return response.entities
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")
            return []

# ==========================================
# Math Pipeline Modules (New/Upgraded)
# ==========================================

class PPRGraphTraverser:
    """
    Replaces ReferenceGraphTraverser.
    Uses Client-side PPR to identify relevant nodes in the graph ROI.
    """
    def __init__(self, neo4j_helper: Neo4jHelper, config: dict):
        self.neo4j = neo4j_helper
        self.ppr_solver = PPRSolver(damping=config.get("Math", {}).get("ppr_damping", 0.85))
        self.hops = config.get("Math", {}).get("subgraph_hops", 2)
        self.limit = config.get("Math", {}).get("subgraph_limit", 2000)

    def calculate_ppr(self, entities: List[Entity]) -> Dict[str, float]:
        """
        1. Query Neo4j to find IDs for the named entities (Anchor Nodes).
        2. Fetch subgraph around anchors.
        3. Run PPR.
        4. Return {node_id: ppr_score}
        """
        # Step 1: Find Anchor IDs
        seed_ids = []
        for ent in entities:
             # Match by name, get ID
            query = "MATCH (n:Level1 {name: $name}) RETURN n.id as id LIMIT 1"
            res = self.neo4j.query(query, {"name": ent.name})
            if res:
                seed_ids.append(res[0]['id'])
        
        if not seed_ids:
            logger.warning(f"No grounded entities found for {entities}. Skipping PPR.")
            return {}
            
        logger.info(f"PPR Seeds: {seed_ids}")
        
        # Step 2: Fetch Subgraph
        subgraph = self.neo4j.get_subgraph(seed_ids, hops=self.hops, limit=self.limit)
        nodes = subgraph.get("nodes", [])
        rels = subgraph.get("relationships", [])
        logger.info(f"Fetched subgraph: {len(nodes)} nodes, {len(rels)} edges.")
        
        if not nodes:
            return {}

        # Step 3: Run PPR
        ppr_scores = self.ppr_solver.run_ppr(nodes, rels, seed_ids)
        
        # Log top 5 high PPR nodes
        top_5 = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top 5 PPR Nodes: {top_5}")
        
        return ppr_scores

class BayesianRetriever:
    """
    Replaces EvidenceRetriever.
    Uses Bayesian scoring to filter retrieval results using PPR scores.
    """
    def __init__(self, qdrant_helper: QdrantHelper, encoder: TransformerEncoder, config: dict):
        self.qdrant = qdrant_helper
        self.encoder = encoder
        self.collection_name = config.get("Retrieval", {}).get("collection_name", "kg_lv1_nodes")
        self.top_k = config.get("Retrieval", {}).get("top_k_dense", 10)
        self.threshold = config.get("Math", {}).get("bayesian_threshold", 0.5)

    def retrieve_and_score(self, query: str, ppr_scores: Dict[str, float]) -> List[Dict]:
        """
        1. Dense Vector Search for candidates.
        2. Filter/Rescore using PPR scores of the retrieved nodes.
           Bayesian Score ~ VectorMatch(Query, Doc) * Prior(Doc in Context | Graph)
           where Prior is derived from PPR.
        """
        try:
            # 1. Vector Search
            vector_np = self.encoder.embed_to_numpy([query]) 
            vector = vector_np[0].tolist()
            
            # Retrieve detailed results
            raw_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=self.top_k, 
                score_threshold=0.0 # Get candidates first, filter later
            )
            
            # 2. Bayesian Rescoring
            scored_results = []
            
            if not ppr_scores:
                # Fallback to pure vector score if PPR failed or empty
                logger.warning("No PPR scores available. Falling back to raw vector scores.")
                max_ppr = 1.0
            else:
                 max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0

            for item in raw_results:
                node_id = item["id"]
                vector_score = item["score"]
                
                # Get PPR score for this node (if it was in the graph ROI)
                # If not in ROI, it's contextually irrelevant -> Low prior
                # But we shouldn't zero it out completely, maybe a small epsilon?
                # Or maybe the graph traversal limit was just too small.
                # Let's use 0.0 for strict graph-constraint or a small epsilon.
                # The paper suggests "Graph ROI Intersection", so if not in graph, should be penalized.
                node_ppr = ppr_scores.get(node_id, 0.0)
                
                # Normalize PPR?
                # Using simple multiplication
                # bayesian_score = vector_score * (1 + alpha * normalized_ppr) ?
                # Or strictly intersection?
                # Let's try: Final = Vector * (0.5 + 0.5 * (PPR / MaxPPR))
                # This ensures pure vector match still counts but graph boosts it.
                
                ppr_factor = 0.5
                if max_ppr > 0:
                     ppr_factor += 0.5 * (node_ppr / max_ppr)
                
                final_score = vector_score * ppr_factor
                
                # Filter
                # We return the object if it passes logic or just top N
                # Let's keep all for now and sort.
                
                item["math_score"] = final_score
                item["ppr_raw"] = node_ppr
                scored_results.append(item)
                
            # Sort by math_score
            scored_results.sort(key=lambda x: x["math_score"], reverse=True)
            
            # Filter top K again or by threshold
            final_selection = [r for r in scored_results if r["math_score"] > 0.0] # Should refine threshold
            
            logger.info(f"Bayesian Scoring: Top 1: {final_selection[0]['name'] if final_selection else 'None'} Score: {final_selection[0]['math_score'] if final_selection else 0}")
            
            return final_selection[:5] # Return top 5 best
            
        except Exception as e:
            logger.error(f"Bayesian Retrieval failed: {e}")
            return []

class Verifier:
    def __init__(self, llm_client, config: dict):
        self.llm = llm_client

    def verify(self, hypothesis_text: str, evidence: List[Dict]) -> Dict:
        # Flatten evidence with PPR info potentially?
        context = "\n".join([f"- {e['payload'].get('name', '')}: {e['payload'].get('definition', '') or 'No definition'} (Conf: {e.get('math_score',0):.2f})" for e in evidence])
        
        prompt = f"""
        Act as a medical judge. Verify if the following statement is clinically valid based on the provided evidence context.
        
        Statement: {hypothesis_text}
        
        Evidence Context (ranked by relevance):
        {context}
        
        Return JSON with 'is_valid' (bool), 'reasoning' (str).
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
        facts = []
        for item in context:
            facts.append(f"Fact: {item['t']}\n- Reasoning: {item['verification']['reasoning']}")
        
        facts_str = "\n".join(facts)
        
        prompt = f"""
        Answer the user query based on the verified facts below.
        
        Query: {query}
        
        Verified Facts:
        {facts_str}
        
        Answer:
        """
        return self.llm.generate(prompt)

# ==========================================
# Main Orchestrator
# ==========================================

class IHDGT_Math_Pipeline:
    def __init__(self, config_path: str):
        logger.info(f"Loading Math Config from {config_path}...")
        self.config = ConfigManager.load_config(config_path)
        
        logger.info("Initializing LLM Client...")
        llm_config = self.config.get("Retrieval", {}).get("LLM", {})
        self.llm = LLMFactory.create_client(llm_config)
        
        logger.info("Connecting to Neo4j...")
        self.neo4j = Neo4jHelper()
        logger.info("Connecting to Qdrant...")
        self.qdrant = QdrantHelper()
        
        logger.info("Loading Transformer Encoder...")
        self.encoder = TransformerEncoder(
            model_name=self.config.get("Encoder", {}).get("model_name"),
            device=self.config.get("Encoder", {}).get("device")
        )
        self.encoder._ensure_model_loaded()
        
        # Modules
        self.decomposer = QueryDecomposer(self.llm, self.config)
        self.linker = EntityLinker(self.llm, self.config)
        
        # MATH MODULES
        self.traverser = PPRGraphTraverser(self.neo4j, self.config)
        self.retriever = BayesianRetriever(self.qdrant, self.encoder, self.config)
        
        self.verifier = Verifier(self.llm, self.config)
        self.responser = Responser(self.llm)
        
        self.concurrency = llm_config.get("concurrency", 5)

    def run(self, query: str):
        logger.info(f">>> START: {query}")
        
        # 1. Decompose
        sub_questions = self.decomposer.decompose(query)
        logger.info(f"Sub-questions: {len(sub_questions)}")
        
        verified_facts = []
        
        def process_sub_q(sq: SubQuestion):
            logger.info(f"Processing: {sq.question}")
            
            # 2. Extract Entities (Seeds)
            entities = self.linker.extract_entities(sq.question)
            
            # 3. Calculate PPR (Math Layer 1)
            # This gives us the "Graph ROI" scores
            ppr_scores = self.traverser.calculate_ppr(entities)
            
            # 4. Bayesian Retrieval (Math Layer 2)
            # Retrieve evidence weighted by PPR
            evidence = self.retriever.retrieve_and_score(sq.question, ppr_scores)
            
            if not evidence:
                logger.warning(f"No evidence found for {sq.question}")
                return None
                
            # 5. Verification
            # Verify the SubQuestion itself against the evidence
            # (Simplification: verifying the question as a statement check)
            check = self.verifier.verify(sq.question, evidence)
            
            if check.get("is_valid", False):
                return {
                    "t": sq.question,
                    "verification": check,
                    "evidence": evidence
                }
            return None

        # Execute Parallel
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [executor.submit(process_sub_q, sq) for sq in sub_questions]
            for future in futures:
                try:
                    res = future.result()
                    if res: verified_facts.append(res)
                except Exception as e:
                    logger.error(f"Error in sub-q: {e}")

        # 6. Synthesis
        answer = self.responser.generate_answer(query, verified_facts)
        
        return answer

    def cleanup(self):
        self.neo4j.close()

def main():
    parser = argparse.ArgumentParser(description="IHD-GT MATH Pipeline")
    parser.add_argument("--query", type=str, required=True, help="Clinical query")
    parser.add_argument("--config", type=str, default="backend/configs/ihdpgt_math_config.yml", help="Path to config")
    
    args = parser.parse_args()
    
    pipeline = None
    try:
        pipeline = IHDGT_Math_Pipeline(args.config)
        answer = pipeline.run(args.query)
        print("\n=== FINAL ANSWER (MATH ENHANCED) ===\n")
        print(answer)
        print("\n====================================\n")
    except Exception as e:
        logger.error(f"Pipeline Execution Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline.cleanup()

if __name__ == "__main__":
    main()
