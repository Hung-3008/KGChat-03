t"""
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


async def initialize_clients():
    """Initialize clients for testing using LLM Registry from config"""
    print("=" * 60)
    print("Initialize clients")
    print("=" * 60)
    clients = {}

    # 1. Load LLM Registry and register providers from config
    print("\nInitializing LLM Registry from config...")
    try:
        # Ensure registry has loaded config and registered providers
        llm_registry.load_config()
        llm_registry.register_all_providers()

        available_providers = llm_registry.get_available_providers()
        default_provider = llm_registry.get_default_provider()
        print(f"LLM Registry initialized successfully")
        print(f"   - Available providers: {', '.join(available_providers)}")
        print(f"   - Default provider: {default_provider}")

        # Get provider from environment variable or use default
        embedding_provider = default_provider
        print(f"   - Using provider for embedding: {embedding_provider}")

    except Exception as e:
        print(f"Error initializing LLM Registry: {e}")
        traceback.print_exc()
        return None

    # 2. Create LLM Client from registry (provider can be changed via config)
    print(
        f"\nCreating LLM Client from registry (provider: {embedding_provider})...")
    try:
        # Get API key from environment if available
        api_key = os.getenv("GEMINI_API_KEY_2") or os.getenv(
            "GEMINI_API_KEY_1")
        model_name = os.getenv("GEMINI_MODEL") or os.getenv("EMBEDDING_MODEL")

        # Create client with overrides from environment
        overrides = {}
        if api_key:
            overrides["api_key"] = api_key
        if model_name:
            overrides["model_name"] = model_name

        llm_client = llm_registry.create_llm_client(
            provider_name=embedding_provider,
            **overrides
        )
        clients["llm_client"] = llm_client
        print(f"LLM Client created from registry")
        print(f"   - Provider: {embedding_provider}")
        if model_name:
            print(f"   - Model: {model_name}")
    except Exception as e:
        print(f"Error creating LLM Client: {e}")
        traceback.print_exc()
        return None

    # 3. Create Embedding Wrapper using LLM Registry
    print("\nCreating Embedding Wrapper from LLM Registry...")
    try:
        # Create a wrapper class to use LLM Registry for embeddings
        class EmbeddingWrapper:
            def __init__(self, llm_registry_instance, provider_name, target_dimension=768, **overrides):
                self.llm_registry = llm_registry_instance
                self.provider_name = provider_name
                self.target_dimension = target_dimension
                self.overrides = overrides

            def _resize_embedding(self, embedding, target_dim):
                """Resize embedding to target dimension by truncating or padding"""
                if len(embedding) == target_dim:
                    return embedding
                elif len(embedding) > target_dim:
                    # Truncate if embedding is larger ( -> 768)
                    return embedding[:target_dim]
                else:
                    # Pad with zeros if embedding is smaller
                    return embedding + [0.0] * (target_dim - len(embedding))

            async def embed(self, texts):
                """Generate embeddings using LLM Registry and resize to target dimension"""
                if isinstance(texts, str):
                    texts = [texts]

                if not texts:
                    return []

                try:
                    # Use LLM Registry to create embeddings
                    # embed_texts() returns List[List[float]]
                    # Wrap in async for interface compatibility
                    import asyncio
                    loop = asyncio.get_event_loop()

                    # Get embedding model from overrides or use default
                    embedding_model = self.overrides.get("embedding_model")

                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.llm_registry.embed_texts(
                            texts=texts,
                            provider_name=self.provider_name,
                            model=embedding_model,
                            **self.overrides
                        )
                    )

                    # Check and resize embeddings to target dimension
                    if embeddings and len(embeddings) > 0:
                        first_dim = len(embeddings[0]) if embeddings[0] else 0

                        if first_dim == 0:
                            print(
                                "Warning: Empty embeddings, creating dummy embeddings")
                            return [[0.0] * self.target_dimension for _ in texts]

                        if first_dim != self.target_dimension:
                            print(
                                f"Resizing embeddings from {first_dim} dimensions to {self.target_dimension} dimensions")
                            embeddings = [self._resize_embedding(emb, self.target_dimension)
                                          for emb in embeddings]
                        else:
                            print(
                                f"Embeddings already have correct {self.target_dimension} dimensions")
                    else:
                        print(
                            "Warning: No embeddings received, creating dummy embeddings")
                        embeddings = [
                            [0.0] * self.target_dimension for _ in texts]

                    return embeddings
                except Exception as e:
                    print(f"Error creating embeddings: {e}")
                    traceback.print_exc()
                    # Create dummy embeddings with target dimension
                    return [[0.0] * self.target_dimension for _ in texts]

        # Create wrapper with default target_dimension of 768
        # Will be updated later if collection has different dimension
        embedding_model = os.getenv("EMBEDDING_MODEL")
        wrapper_overrides = {}
        if embedding_model:
            wrapper_overrides["embedding_model"] = embedding_model

        clients["ollama_client"] = EmbeddingWrapper(
            llm_registry_instance=llm_registry,
            provider_name=embedding_provider,
            target_dimension=768,
            **wrapper_overrides
        )
        print("Embedding Wrapper created from LLM Registry")
        print(f"   - Provider: {embedding_provider}")
        print(f"   - Default using 768 dimensions for embeddings")
        if embedding_model:
            print(f"   - Embedding model: {embedding_model}")
    except Exception as e:
        print(f"Error creating Embedding Wrapper: {e}")
        traceback.print_exc()
        return None

    # 4. Qdrant Client (use QdrantClient directly because triple_level_retriever uses query_points)
    print("\nInitializing Qdrant Client...")
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        print(f"   - Connecting to Qdrant: {qdrant_host}:{qdrant_port}")

        qdrant_client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port
        )

        # Test connection by getting collections
        print("   - Checking connection...")
        try:
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections] if hasattr(
                collections, 'collections') else []
            print(f"Qdrant Client connected successfully")
            print(f"   - Host: {qdrant_host}:{qdrant_port}")
            print(f"   - Available collections: {collection_names}")
            print(f"   - Total collections: {len(collection_names)}")
        except Exception as conn_e:
            print(f"Error checking Qdrant connection: {conn_e}")
            print(
                f"Warning: Qdrant may not be running or not accessible at {qdrant_host}:{qdrant_port}")
            print("Check:")
            print("   1. Qdrant is running: docker ps | grep qdrant")
            print("   2. Correct port: docker port <qdrant_container>")
            print("   3. Firewall/network is not blocking connection")
            raise

        # Check collection "kg_level1"
        collection_dimension = None
        if "kg_level1" not in collection_names:
            print("\nWarning: Collection 'kg_level1' does not exist in Qdrant")
            print("Need to create collection and import data into Qdrant")
            print(
                f"   - Current collections: {collection_names if collection_names else 'None'}")
        else:
            print(f"\nCollection 'kg_level1' exists")
            # Check number of points and vector dimension in collection
            try:
                collection_info = qdrant_client.get_collection("kg_level1")
                # Use points_count instead of vectors_count
                points_count = 0
                if hasattr(collection_info, 'points_count'):
                    points_count = collection_info.points_count or 0
                elif hasattr(collection_info, 'vectors_count'):
                    points_count = collection_info.vectors_count or 0
                print(f"   - Collection 'kg_level1':")
                print(f"     + Number of points: {points_count}")
                if points_count == 0:
                    print("     Warning: Collection is empty - need to import data")

                # Get vector dimension from collection config
                try:
                    # Try multiple ways to get vector dimension
                    if hasattr(collection_info, 'config'):
                        config = collection_info.config
                        if hasattr(config, 'params'):
                            params = config.params
                            # Method 1: Direct vectors config
                            if hasattr(params, 'vectors'):
                                vectors = params.vectors
                                if hasattr(vectors, 'size'):
                                    collection_dimension = vectors.size
                                elif isinstance(vectors, dict):
                                    # Named vectors
                                    for vec_name, vec_config in vectors.items():
                                        if hasattr(vec_config, 'size'):
                                            collection_dimension = vec_config.size
                                            break
                            # Method 2: Check config directly
                            if not collection_dimension and hasattr(params, 'vector_size'):
                                collection_dimension = params.vector_size

                    # If still not found, try getting from collection_info directly
                    if not collection_dimension:
                        # Try other attributes
                        if hasattr(collection_info, 'vector_size'):
                            collection_dimension = collection_info.vector_size
                        elif hasattr(collection_info, 'config') and hasattr(collection_info.config, 'vector_size'):
                            collection_dimension = collection_info.config.vector_size

                    if collection_dimension:
                        print(f"   - Vector dimension: {collection_dimension}")
                    else:
                        print(
                            "   Warning: Cannot determine vector dimension, using default 768")
                        collection_dimension = 768  # Default on error
                except Exception as dim_e:
                    print(
                        f"   Warning: Error getting vector dimension: {dim_e}")
                    collection_dimension = 768  # Default on error
            except Exception as e:
                print(f"   Warning: Cannot get collection info: {e}")

        # Update target_dimension for EmbeddingWrapper if collection has dimension other than 768
        # If no collection_dimension, keep 768 (default)
        if collection_dimension and collection_dimension != 768:
            if clients.get("ollama_client"):
                clients["ollama_client"].target_dimension = collection_dimension
                print(
                    f"   Updated embedding target dimension: {collection_dimension}")
        elif collection_dimension == 768:
            print("   Collection dimension (768) matches default")
        else:
            print("   Using default vector dimension: 768")

        clients["qdrant_client"] = qdrant_client
        print("\nQdrant Client initialized and verified successfully")
    except Exception as e:
        print(f"\nError initializing Qdrant Client: {e}")
        print("Ensure Qdrant is running:")
        print("   - Docker: docker compose up -d qdrant")
        print("   - Or: docker run -p 6333:6333 qdrant/qdrant")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        print(
            f"   - Check health: curl http://{qdrant_host}:{qdrant_port}/health")
        print(
            f"   - Check collections: curl http://{qdrant_host}:{qdrant_port}/collections")
        import traceback
        traceback.print_exc()
        return None

    # 5. Neo4j Client
    print("\nInitializing Neo4j Client...")
    if Neo4jClient is None:
        print("Warning: Neo4jClient not available")
        print("Warning: Skipping Neo4j tests")
        clients["neo4j_client"] = None
    else:
        try:
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER_NAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "123456789")

            # Initialize Neo4jClient similar to check_retrival_qdrant2neo4j.py
            neo4j_client = Neo4jClient(
                uri=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password
            )
            print(f"Created Neo4jClient for {neo4j_uri}")

            # Verify connectivity before testing
            try:
                connected = await neo4j_client.verify_connectivity()
                if connected:
                    print("Neo4j connectivity verified")
                else:
                    print("Warning: Neo4j connectivity verification failed")
            except Exception as verify_e:
                print(f"Warning: Error verifying connectivity: {verify_e}")

            # Test connection by trying a simple query
            try:
                test_result = await neo4j_client.execute_query("RETURN 1 as test")
                if test_result:
                    print("Neo4j connection test successful")

                # Check number of Level1 nodes in database
                count_query = "MATCH (n:Level1) RETURN count(n) as total"
                count_result = await neo4j_client.execute_query(count_query)
                total_nodes = count_result[0].get(
                    'total', 0) if count_result and len(count_result) > 0 else 0
                print(
                    f"   - Number of Level1 nodes in database: {total_nodes}")

                if total_nodes == 0:
                    print(
                        "   Warning: Database is empty! Data may be in a different database.")
                    # Try querying with different database
                    print("   Check if there is a database name issue...")
                else:
                    print(f"   Database has {total_nodes} Level1 nodes")
            except Exception as test_e:
                print(f"Warning: Neo4j connection test failed: {test_e}")
                print("Warning: Continuing but may have errors when using Neo4j")
                import traceback
                traceback.print_exc()

            clients["neo4j_client"] = neo4j_client
            print("Neo4j Client initialized")
        except Exception as e:
            print(f"Error initializing Neo4j Client: {e}")
            print("Warning: Ensure Neo4j is running and credentials are correct")
            clients["neo4j_client"] = None

    return clients


async def test_retrieve_from_knowledge_graph(clients):
    """Test retrieve_from_knowledge_graph function"""
    print("\n" + "=" * 60)
    print("TEST: retrieve_from_knowledge_graph")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("Warning: Neo4j client not available, skipping this test")
        return None

    if not clients.get("qdrant_client"):
        print("Warning: Qdrant client not available, skipping this test")
        return None

    # Test query and keywords
    test_query = "What are the symptoms and treatment of diabetes?"
    high_keywords = ["diabetes", "symptoms", "treatment"]
    low_keywords = ["blood sugar", "insulin", "glucose", "cathepsin L"]

    print(f"\nQuery: {test_query}")
    print(f"High-level keywords: {high_keywords}")
    print(f"Low-level keywords: {low_keywords}")

    try:
        result = await retrieve_from_knowledge_graph(
            high_level_keywords=high_keywords,
            low_level_keywords=low_keywords,
            neo4j_client=clients["neo4j_client"],
            ollama_client=clients["ollama_client"],
            qdrant_client=clients["qdrant_client"],
            top_k=5,
            similarity_threshold=0.7
        )

        print("\nRetrieval results:")
        l1_nodes = len(result.get('level1_nodes', []))
        l2_nodes = len(result.get('level2_nodes', []))
        relationships = len(result.get('relationships', []))

        print(f"   - Level 1 nodes: {l1_nodes}")
        print(f"   - Level 2 nodes: {l2_nodes}")
        print(f"   - Relationships: {relationships}")

        # Warning if no results
        if l1_nodes == 0:
            print("\nWarning: NO NODES FOUND - Possible reasons:")
            print("   1. Qdrant collection 'kg_level1' has no data")
            print("   2. Neo4j database has no corresponding data")
            print("   3. Embeddings do not match data in Qdrant")
            print("   4. Similarity threshold too high (current: 0.7)")
            print("   5. Entity IDs in Qdrant payload do not match Neo4j")
            print("\nTo test fully, you need:")
            print("   - Import data into Qdrant (collection 'kg_level1')")
            print("   - Import corresponding data into Neo4j")
            print(
                "   - Ensure 'node_id' in Qdrant payload matches 'id' in Neo4j")
            print(
                "   - Ensure embeddings in Qdrant were created from the same model as query embeddings")

        # Display some sample nodes
        if result.get('level1_nodes'):
            print("\nSample Level 1 nodes:")
            for i, node in enumerate(result['level1_nodes'][:3], 1):
                print(
                    f"   {i}. {node.get('name', 'Unknown')} ({node.get('entity_type', 'Unknown')}) - Score: {node.get('similarity_score', 0):.4f}")

        if result.get('level2_nodes'):
            print("\nSample Level 2 nodes:")
            for i, node in enumerate(result['level2_nodes'][:3], 1):
                cui = node.get('cui', '')
                cui_str = f" (CUI: {cui})" if cui else ""
                semantic_types = node.get('semantic_types', [])
                entity_type = semantic_types[0] if semantic_types else node.get(
                    'entity_type', 'Unknown')
                print(
                    f"   {i}. {node.get('name', 'Unknown')}{cui_str} ({entity_type})")

        return result

    except Exception as e:
        print(f"\nError testing retrieve_from_knowledge_graph: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_retrieve_level1_nodes(clients):
    """Test retrieve_level1_nodes function"""
    print("\n" + "=" * 60)
    print("TEST: retrieve_level1_nodes")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("Warning: Neo4j client not available, skipping this test")
        return None

    if not clients.get("qdrant_client"):
        print("Warning: Qdrant client not available, skipping this test")
        return None

    # Create test embeddings
    test_keywords = ["diabetes", "treatment"]
    print(f"\nTest keywords: {test_keywords}")

    try:
        # Generate embeddings using Gemini wrapper
        embeddings = await clients["ollama_client"].embed(test_keywords)
        print(f"Created {len(embeddings)} embeddings using Gemini")

        # Retrieve Level 1 nodes
        # Note: retrieve_level1_nodes only accepts top_k and similarity_threshold
        nodes = await retrieve_level1_nodes(
            embeddings=embeddings,
            qdrant_client=clients["qdrant_client"],
            neo4j_client=clients["neo4j_client"],
            top_k=5,
            similarity_threshold=0.7
        )

        num_nodes = len(nodes)
        print(f"\nRetrieved {num_nodes} Level 1 nodes")

        if num_nodes == 0:
            print("\nWarning: NO NODES FOUND - Possible reasons:")
            print("   1. Qdrant collection 'kg_level1' has no data")
            print("   2. Neo4j database has no corresponding data")
            print("   3. Embeddings do not match data in Qdrant")
            print("   4. Similarity threshold too high (current: 0.7)")
            print(
                "   5. Entity IDs in Qdrant payload do not match Neo4j 'id' field")
        elif nodes:
            print("\nSample nodes:")
            for i, node in enumerate(nodes[:3], 1):
                print(
                    f"   {i}. {node.get('name', 'Unknown')} (score: {node.get('similarity_score', 0):.4f})")

        return nodes

    except Exception as e:
        print(f"\nError testing retrieve_level1_nodes: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_retrieve_level2_references(clients, level1_nodes):
    """Test retrieve_level2_references function"""
    print("\n" + "=" * 60)
    print("TEST: retrieve_level2_references")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("Warning: Neo4j client not available, skipping this test")
        return None, None

    if not level1_nodes:
        print("Warning: No Level 1 nodes to test, skipping this test")
        return None, None

    try:
        # Get a few Level 1 nodes to test
        test_level1_nodes = level1_nodes[:3] if len(
            level1_nodes) >= 3 else level1_nodes
        print(f"\nTesting with {len(test_level1_nodes)} Level 1 nodes")

        level2_nodes, relationships = await retrieve_level2_references(
            level1_nodes=test_level1_nodes,
            neo4j_client=clients["neo4j_client"],
            max_references=5
        )

        print(f"\nRetrieved:")
        print(f"   - Level 2 nodes: {len(level2_nodes)}")
        print(f"   - Relationships: {len(relationships)}")

        if level2_nodes:
            print("\nSample Level 2 nodes:")
            for i, node in enumerate(level2_nodes[:3], 1):
                cui = node.get('cui', '')
                cui_str = f" (CUI: {cui})" if cui else ""
                semantic_types = node.get('semantic_types', [])
                entity_type = semantic_types[0] if semantic_types else node.get(
                    'entity_type', 'Unknown')
                definition = node.get('definition', '')
                # Format definition string
                if definition:
                    if len(definition) > 50:
                        def_str = f": {definition[:50]}..."
                    else:
                        def_str = f": {definition}"
                else:
                    def_str = ""
                print(
                    f"   {i}. {node.get('name', 'Unknown')}{cui_str} ({entity_type}){def_str}")

        return level2_nodes, relationships

    except Exception as e:
        print(f"\nError testing retrieve_level2_references: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_retrieve_level3_references(clients, level2_nodes):
    """Test retrieve_level3_references function - query from Level2 to Level3"""
    print("\n" + "=" * 60)
    print("TEST: retrieve_level3_references")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("Warning: Neo4j client not available, skipping this test")
        return None, None

    if not level2_nodes:
        print("Warning: No Level 2 nodes to test, skipping this test")
        print("Trying to get Level2 nodes directly from database...")

        # Try to get Level2 nodes directly from database
        try:
            query = """
            MATCH (l2:Level2)
            WHERE l2.cui IS NOT NULL AND l2.cui <> ''
            RETURN l2.id AS id, l2.name AS name, l2.cui AS cui, 
                   l2.definition AS definition, l2.semantic_types AS semantic_types
            LIMIT 5
            """
            results = await clients["neo4j_client"].execute_query(query)

            if results and len(results) > 0:
                level2_nodes = []
                for record in results:
                    semantic_types = record.get("semantic_types", [])
                    entity_type = semantic_types[0] if semantic_types else "CONCEPT"
                    level2_data = {
                        "id": record.get("id", ""),
                        "entity_id": record.get("id", ""),
                        "name": record.get("name", "Unknown"),
                        "cui": record.get("cui", ""),
                        "definition": record.get("definition", ""),
                        "semantic_types": semantic_types,
                        "entity_type": entity_type,
                        "description": record.get("definition", "")
                    }
                    level2_nodes.append(level2_data)
                print(
                    f"Retrieved {len(level2_nodes)} Level2 nodes from database")
            else:
                print("Warning: No Level2 nodes with CUI found in database")
                return None, None
        except Exception as e:
            print(f"Warning: Error getting Level2 nodes from database: {e}")
            return None, None

    try:
        # Get a few Level 2 nodes to test
        test_level2_nodes = level2_nodes[:5] if len(
            level2_nodes) >= 5 else level2_nodes
        print(f"\nTesting with {len(test_level2_nodes)} Level 2 nodes")

        # Display Level2 nodes being tested
        print("\nLevel 2 nodes being tested:")
        for i, node in enumerate(test_level2_nodes, 1):
            cui = node.get('cui', '')
            print(f"   {i}. {node.get('name', 'Unknown')} (CUI: {cui})")

        level3_nodes, relationships = await retrieve_level3_references(
            level2_nodes=test_level2_nodes,
            neo4j_client=clients["neo4j_client"],
            max_references=10  # Get more to see full patient examples
        )

        print(f"\nRetrieved:")
        print(f"   - Level 3 nodes: {len(level3_nodes)}")
        print(f"   - Relationships: {len(relationships)}")

        if level3_nodes:
            print("\n" + "=" * 60)
            print("ALL LEVEL 3 NODES INFORMATION (PATIENT EXAMPLES)")
            print("=" * 60)

            import json

            for i, node in enumerate(level3_nodes, 1):
                print(f"\n{'=' * 60}")
                print(f"--- Level 3 Node #{i} ---")
                print(f"{'=' * 60}")

                # Basic Information
                print(f"\nBasic Information:")
                print(f"   ID: {node.get('id', 'N/A')}")
                print(
                    f"   Level2 Node ID (CUI): {node.get('level2_node_id', 'N/A')}")
                print(f"   Subject ID: {node.get('subject_id', 'N/A')}")
                print(f"   Source: {node.get('source', 'N/A')}")

                # Patient Demographics
                print(f"\nPatient Demographics:")
                print(f"   Gender: {node.get('gender', 'N/A')}")
                print(f"   Age: {node.get('anchor_age', 'N/A')}")
                print(f"   Anchor Year: {node.get('anchor_year', 'N/A')}")
                print(
                    f"   Anchor Year Group: {node.get('anchor_year_group', 'N/A')}")

                # Admission Information
                admission_info = node.get('admission_info_json')
                if admission_info:
                    print(f"\nAdmission Information:")
                    if isinstance(admission_info, str):
                        try:
                            admission_info = json.loads(admission_info)
                        except:
                            pass
                    if isinstance(admission_info, dict):
                        for key, value in admission_info.items():
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {admission_info}")
                else:
                    print(f"\nAdmission Information: N/A")

                # Procedures
                procedures = node.get('procedures_json')
                if procedures:
                    print(
                        f"\nProcedures (Total: {node.get('total_procedures_count', 0)}):")
                    if isinstance(procedures, str):
                        try:
                            procedures = json.loads(procedures)
                        except:
                            pass
                    if isinstance(procedures, list):
                        for j, proc in enumerate(procedures, 1):
                            if isinstance(proc, dict):
                                print(f"   Procedure #{j}:")
                                for key, value in proc.items():
                                    print(f"      {key}: {value}")
                            else:
                                print(f"   Procedure #{j}: {proc}")
                    else:
                        print(f"   {procedures}")
                else:
                    print(
                        f"\nProcedures: None (Total: {node.get('total_procedures_count', 0)})")

                # Medications
                medications = node.get('medications_json')
                if medications:
                    print(f"\nMedications:")
                    if isinstance(medications, str):
                        try:
                            medications = json.loads(medications)
                        except:
                            pass
                    if isinstance(medications, list):
                        if len(medications) > 0:
                            for j, med in enumerate(medications, 1):
                                if isinstance(med, dict):
                                    print(f"   Medication #{j}:")
                                    for key, value in med.items():
                                        print(f"      {key}: {value}")
                                else:
                                    print(f"   Medication #{j}: {med}")
                        else:
                            print(f"   None")
                    else:
                        print(f"   {medications}")
                else:
                    print(f"\nMedications: None")

                # Diagnoses
                diagnoses = node.get('relevant_diagnoses_json')
                if diagnoses:
                    print(
                        f"\nRelevant Diagnoses (Total: {node.get('total_diagnoses_count', 0)}):")
                    if isinstance(diagnoses, str):
                        try:
                            diagnoses = json.loads(diagnoses)
                        except:
                            pass
                    if isinstance(diagnoses, list):
                        for j, diag in enumerate(diagnoses, 1):
                            if isinstance(diag, dict):
                                print(f"   Diagnosis #{j}:")
                                for key, value in diag.items():
                                    print(f"      {key}: {value}")
                            else:
                                print(f"   Diagnosis #{j}: {diag}")
                    else:
                        print(f"   {diagnoses}")
                else:
                    print(
                        f"\nRelevant Diagnoses: None (Total: {node.get('total_diagnoses_count', 0)})")

                # Services
                services = node.get('services_json')
                if services:
                    print(
                        f"\nServices (Total: {node.get('total_services_count', 'N/A')}):")
                    if isinstance(services, str):
                        try:
                            services = json.loads(services)
                        except:
                            pass
                    if isinstance(services, list):
                        for j, svc in enumerate(services, 1):
                            if isinstance(svc, dict):
                                print(f"   Service #{j}:")
                                for key, value in svc.items():
                                    print(f"      {key}: {value}")
                            else:
                                print(f"   Service #{j}: {svc}")
                    else:
                        print(f"   {services}")
                else:
                    print(
                        f"\nServices: None (Total: {node.get('total_services_count', 'N/A')})")

                # Summary counts
                print(f"\nSummary Counts:")
                print(
                    f"   Total Diagnoses: {node.get('total_diagnoses_count', 0)}")
                print(
                    f"   Total Procedures: {node.get('total_procedures_count', 0)}")
                print(
                    f"   Total Services: {node.get('total_services_count', 'N/A')}")
        else:
            print("\nWarning: NO LEVEL 3 NODES FOUND - Possible reasons:")
            print("   1. Level2 nodes do not have CUI")
            print("   2. No Level3 nodes have level2_node_id matching Level2 CUI")
            print("   3. Database has no Level3 data")
            print("\nTo test fully, you need:")
            print("   - Ensure Level2 nodes have CUI")
            print("   - Ensure Level3 nodes have level2_node_id matching Level2 CUI")

        return level3_nodes, relationships

    except Exception as e:
        print(f"\nError testing retrieve_level3_references: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_format_functions(level1_nodes, level2_nodes, relationships):
    """Test format functions"""
    print("\n" + "=" * 60)
    print("TEST: Format Functions")
    print("=" * 60)

    if not level1_nodes:
        print("Warning: No nodes to test, skipping this test")
        return

    try:
        # Test format_retrieval_results
        print("\nTesting format_retrieval_results...")
        formatted_text = format_retrieval_results(
            level1_nodes=level1_nodes[:5] if len(
                level1_nodes) >= 5 else level1_nodes,
            level2_nodes=level2_nodes[:5] if level2_nodes and len(
                level2_nodes) >= 5 else (level2_nodes or [])[:5],
            relationships=relationships[:5] if relationships and len(
                relationships) >= 5 else (relationships or [])[:5]
        )
        print(f"Formatted text length: {len(formatted_text)} characters")
        preview = formatted_text[:500] if len(
            formatted_text) > 500 else formatted_text
        print(f"Preview (first 500 chars):\n{preview}...")

        # Display additional information about format
        print("\nFormatted text information:")
        print(f"   - Total characters: {len(formatted_text)}")
        print(f"   - Number of lines: {len(formatted_text.split(chr(10)))}")

        # Check main sections
        if "# KEY CONCEPTS" in formatted_text:
            print("   Has KEY CONCEPTS section")
        if "# RELATIONSHIPS" in formatted_text:
            print("   Has RELATIONSHIPS section")
        if "# DETAILED INFORMATION" in formatted_text:
            print("   Has DETAILED INFORMATION section")

    except Exception as e:
        print(f"\nError testing format functions: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function to run all tests"""
    print("\n" + "=" * 60)
    print("STARTING TEST DUAL LEVEL RETRIEVER")
    print("=" * 60)

    # Initialize clients
    clients = await initialize_clients()
    if not clients:
        print("\nError: Cannot initialize clients. Please check configuration.")
        return

    # Test 1: retrieve_from_knowledge_graph
    kg_result = await test_retrieve_from_knowledge_graph(clients)

    # Test 2: retrieve_level1_nodes (if test 1 succeeded)
    level1_nodes = None
    if kg_result:
        level1_nodes = kg_result.get('level1_nodes', [])

    # If no nodes from test 1, test retrieve_level1_nodes separately
    if not level1_nodes:
        level1_nodes = await test_retrieve_level1_nodes(clients)

    # Test 3: retrieve_level2_references
    level2_nodes, relationships = await test_retrieve_level2_references(clients, level1_nodes)

    # Test 4: retrieve_level3_references (query from Level2 to Level3)
    # Print all Level3 nodes information (patient examples) in test function
    await test_retrieve_level3_references(clients, level2_nodes)

    # Test 5: Format functions
    await test_format_functions(level1_nodes, level2_nodes, relationships)

    # Close connections
    print("\n" + "=" * 60)
    print("CLOSING CONNECTIONS")
    print("=" * 60)
    try:
        if clients.get("neo4j_client"):
            # Check if close method exists
            if hasattr(clients["neo4j_client"], "close"):
                await clients["neo4j_client"].close()
                print("Neo4j connection closed")
            else:
                print("Neo4j client does not have close() method")
    except Exception as e:
        print(f"Warning: Error closing Neo4j connection: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)

    # Summary of results
    print("\nSUMMARY:")
    if kg_result:
        total_nodes = (len(kg_result.get('level1_nodes', [])) +
                       len(kg_result.get('level2_nodes', [])))
        if total_nodes == 0:
            print("Warning: No nodes retrieved")
            print("To test fully, you need:")
            print("   - Import data into Qdrant (collection 'kg_level1')")
            print("   - Import corresponding data into Neo4j")
            print(
                "   - Ensure 'node_id' in Qdrant matches 'id' in Neo4j")
        else:
            print(f"Retrieved {total_nodes} nodes from knowledge graph")
            print(f"   - Level 1: {len(kg_result.get('level1_nodes', []))}")
            print(f"   - Level 2: {len(kg_result.get('level2_nodes', []))}")
    else:
        print(
            "Warning: Cannot test retrieval - may be due to Qdrant or Neo4j having no data")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
