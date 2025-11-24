import os
import yaml
import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

from backend.llm.ollama_client import OllamaClient
from backend.llm.gemini_client import GeminiClient
from backend.encoders.transformer_encoder import TransformerEncoder
from backend.utils.qdrant_helper import QdrantHelper
from backend.utils.neo4j_helper import Neo4jHelper
from backend.query.prompts import KEYWORD_EXTRACTION_PROMPT, ANSWER_GENERATION_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RetrievalManager")

class Keywords(BaseModel):
    keywords: List[str]

class RetrievalManager:
    def __init__(self, config_path: str = "backend/configs/configs.yml"):
        self.config = self._load_config(config_path)
        
        # Initialize clients
        self.ollama_client = OllamaClient(self.config.get("LLM", {}))
        self.gemini_client = GeminiClient(self.config.get("Gemini", {})) 
        self.encoder = TransformerEncoder(
            model_name=self.config.get("Encoder", {}).get("model_name"),
            device=self.config.get("Encoder", {}).get("device")
        )
        self.qdrant = QdrantHelper()
        try:
            self.neo4j = Neo4jHelper()
        except Exception as e:
            logger.warning(f"Neo4j initialization failed: {e}. Graph retrieval will be disabled.")
            self.neo4j = None
        
        logger.info("RetrievalManager initialized successfully.")

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise

    def extract_keywords(self, question: str) -> List[str]:
        logger.info("Extracting keywords...")
        prompt = KEYWORD_EXTRACTION_PROMPT.format(question=question)
        try:
            response = self.ollama_client.generate(prompt, format=Keywords)
            if isinstance(response, dict) and "keywords" in response:
                keywords = response["keywords"]
            else: 
                keywords = [k.strip() for k in str(response).split(",")]
            
            logger.info(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        logger.info(f"Embedding text: {text}")
        try:
            tensor = self.encoder.embed([text])
            vector = tensor[0].tolist()
            return vector
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def search_qdrant(self, vector: List[float], collection_name: str = "kg_lv1_nodes", top_k: int = 5) -> List[str]:
        logger.info(f"Searching Qdrant collection '{collection_name}'...")
        
        node_ids = self.qdrant.search(
            collection_name=collection_name, 
            query_vector=vector, 
            limit=top_k,
            score_threshold=0.65
        )
        logger.info(f"Found {len(node_ids)} nodes in Qdrant: {node_ids}")
        return node_ids

    def get_neo4j_data(self, node_ids: List[str]) -> List[Dict]:
        logger.info(f"Querying Neo4j for {len(node_ids)} nodes...")
        if not node_ids:
            return []
        
        query = """
        MATCH (n:Level1)-[r]-(m)
        WHERE n.id IN $node_ids
        RETURN n.name AS source, n.semantic_type AS source_type, type(r) AS relation, m.name AS target, m.semantic_type AS target_type
        """
        if not self.neo4j:
            logger.warning("Neo4j is not initialized. Returning empty graph data.")
            return []

        try:
            results = self.neo4j.query(query, parameters={"node_ids": node_ids})
            logger.info(f"Retrieved {len(results)} relationships from Neo4j.")
            return results
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []

    def format_graph_context(self, data: List[Dict]) -> str:
        logger.info("Formatting graph context...")
        if not data:
            return "No graph data found."
        
        lines = []
        for item in data:
            source = item.get("source", "Unknown")
            source_type = item.get("source_type", "Entity")
            relation = item.get("relation", "related_to")
            target = item.get("target", "Unknown")
            target_type = item.get("target_type", "Entity")
            
            line = f"The {source_type} '{source}' has relationship '{relation}' with {target_type} '{target}'."
            lines.append(line)
        
        context = "\n".join(lines)
        return context

    def get_google_grounding(self, question: str) -> str:
        logger.info("Getting Google Grounding info...")
        try:
            response = self.gemini_client.generate(question, grounding=True)
            return str(response)
        except Exception as e:
            logger.error(f"Google Grounding failed: {e}")
            return ""

    def generate_answer(self, question: str, graph_context: str, google_context: str) -> str:
        logger.info("Generating final answer...")
        
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=question,
            graph_context=graph_context,
            google_context=google_context
        )
        
        try:
            response = self.ollama_client.generate(prompt)
            return str(response)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Failed to generate answer."

    def run(self, question: str, grounding: bool = False) -> str:
        logger.info(f"Starting retrieval process for question: '{question}'")
        
        # 1. Extract Keywords
        keywords = self.extract_keywords(question)
        if not keywords:
            logger.warning("No keywords extracted. Using question as keyword.")
            keywords = [question]
            
        text_to_embed = " ".join(keywords)
        vector = self.get_embedding(text_to_embed)
        
        # 3. Vector Search
        node_ids = []
        if vector:
            node_ids = self.search_qdrant(vector)
        
        # 4. Get Neo4j Data
        graph_data = self.get_neo4j_data(node_ids)
        graph_context = self.format_graph_context(graph_data)
        
        # 5. Google Grounding
        if grounding:
            google_context = self.get_google_grounding(question)
        else:
            google_context = ""
        
        # 6. Generate Answer
        answer = self.generate_answer(question, graph_context, google_context)
        
        logger.info("Retrieval process completed.")
        return answer

    def run_for_frontend(self, question: str, grounding: bool = True) -> Dict[str, Any]:
        logger.info(f"Starting frontend retrieval process for question: '{question}'")
        
        steps_data = {
            "question": question,
            "keywords": [],
            "qdrant_nodes": [],
            "graph_data": [],
            "google_grounding": "",
            "final_answer": ""
        }
        
        # 1. Extract Keywords
        keywords = self.extract_keywords(question)
        if not keywords:
            logger.warning("No keywords extracted. Using question as keyword.")
            keywords = [question]
        steps_data["keywords"] = keywords
            
        text_to_embed = " ".join(keywords)
        vector = self.get_embedding(text_to_embed)
        
        # 3. Vector Search
        node_ids = []
        if vector:
            node_ids = self.search_qdrant(vector)
        steps_data["qdrant_nodes"] = node_ids
        
        # 4. Get Neo4j Data
        graph_data = self.get_neo4j_data(node_ids)
        steps_data["graph_data"] = graph_data
        graph_context = self.format_graph_context(graph_data)
        
        # 5. Google Grounding
        if grounding:
            google_context = self.get_google_grounding(question)
            steps_data["google_grounding"] = google_context
        else:
            google_context = ""
        
        # 6. Generate Answer
        answer = self.generate_answer(question, graph_context, google_context)
        steps_data["final_answer"] = answer
        
        logger.info("Frontend retrieval process completed.")
        return steps_data

if __name__ == "__main__":
    try:
        manager = RetrievalManager()
        test_question = "What are the treatments for diabetes?"
        print(f"\n--- Testing with question: {test_question} ---\n")
        final_answer = manager.run(test_question)
        print(f"\n--- Final Answer ---\n{final_answer}\n")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
