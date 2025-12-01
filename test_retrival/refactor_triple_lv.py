"""
Test triple level retriever 
"""
import logging
import types
import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import traceback
# Load enviroment variables
load_dotenv()

# root path and backend path
root_path = Path(__file__).parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(backend_path))

try:
    from backend.llm.factory import llm_registry
    from qdrant_client import QdrantClient
    from backend.db.neo4j_client import Neo4jClient
    from backend.db.vector_db import VectorDBClient
    from backend.retrieval.triple_level_retriever import (
        retrieve_from_knowledge_graph,
        retrieve_level1_nodes,
        retrieve_level2_references,
        retrieve_level3_references,
        format_retrieval_results
    )
    from backend.utils.logging import get_logger
    from backend.pipeline.pipeline_prompts import PROMPTS
except ImportError as e:
    print(f"Error import modules: {e}")
    traceback.print_exc()
    sys.exit(1)


async def intitialize_clients():
    """Initialize clients for testing using LLM Registry form config """
    print("=" * 60)
    print("Initialize clients")
    print("=" * 60)
    clients = {}
    print("\n Create LLM Registry from config")
    try:
        llm_registry.load_config()
        llm_registry.register_all_providers()
        available_providers = llm_registry.get_available_providers()
        default_provider = llm_registry.get_default_provider()
        print(f"LLM Registry from config created successful")
        print(f"Available providers: {', '.join(available_providers)}")
        print(f"Default provider: {default_provider}")

        embedding_provider = default_provider
        print(f"Using default provider for embedding: {embedding_provider}")
    except Exception as e:
        print(f"Error create LLM Registry from config: {e}")
        traceback.print_exc()
        return None
    print(
        f"\n Create LLM Client from registry (provider: {embedding_provider})...")

    try:
        api_key = os.getenv("GEMINI_API_KEY_2") or os.getenv(
            "GEMINI_API_KEY_3")
        model_name = os.getenv("GEMINI_MODEL") or os.getenv("EMBEDDING_MODEL")

        # Create client with overrides from environment
        overrides = {}
        if api_key:
            overrides['api_key'] = api_key
        if model_name:
            overrides['model_name'] = model_name
        llm_client = llm_registry.create_llm_client(
            provider_name=embedding_provider,
            **overrides
        )
        clients['llm_client'] = llm_client
        print(f"LLM Client from registry created successfully")
        print(f" - Provider: {embedding_provider}")
        if model_name:
            print(f" - Model: {model_name}")
    except Exception as e:
        print(f"Error create LLM Client:{e}")
        traceback.print_exc()
        return None
    # 3. Create Embedding Wrapper using LLM Registry
    print("\n Create Embedding Wrapper using LLM Registry...")
    try:
        class EmbeddingWrapper:
            def __init__(self, llm_registry_instance, provider_name, target_dimension=768, **overriders):
                self.llm_registry = llm_registry_instance
                sefl.provider_name = provider_name
                self.target_dimension = target_dimension
                self.overrides = overrides

            def _resize_embedding(self, embedding, target_dim):
                """Resize embedding to target dimension by truncating or padding"""
                if len(embedding) == target_dim:
                    return embedding

                elif len(embedding) > target_dim:
                    return embedding[:target_dim]
                else:
                    return embedding + [0.0]*(target_dim - len(embedding))

            async def embed(self, texts):
                """Generate embeddings using LLM Registry and resize to target dimension"""
                if isinstance(texts, str):
                    texts = [texts]

                if not texts:
                    return []
                try:
                    # embed_texts() returns List[List[float]]
                    import asyncio
                    loop = asyncio.get_event_loop()
                    embeddings_model = self.overrides.get("embedding_model")
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.llm_registry.embed_texts(
                            texts=texts,
                            provider_name=self.provider_name,
                            model=embeddings_model,
                            **self.overrides
                        )
                    )
                    # check and resize embeddings to target dimension
                    if embeddings and len(embeddings) > 0:
                        first_dim = len(embeddings[0]) if embeddings[0] else 0
                        if first_dim == 0:
                            print("Empty embeddings, create dummy embeddings")
                            return [[0.0] * self.target_dimension for _ in texts]
                        if first_dim != self.target_dimension:
                            print(
                                f"Resizing embeddings from {first_dim} dimensions to {self.target_dimension} dimension")
                            embeddings = [self._resize_embedding(
                                emb, self.target_dimension) for emb in embeddings]
                        else:
                            print(
                                f"Embedings already have correct {self.target_dimension} dimensions")
                    else:
                        print("No embeddings received, create dummy embeddings")
