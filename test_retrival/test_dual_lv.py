"""
Script test để kiểm tra Dual Level Retriever
Test các chức năng chính của dual_level_retriever.py
"""
import logging
import types
import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Thêm cả thư mục gốc và backend vào path
root_path = Path(__file__).parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(backend_path))

# Import các modules cơ bản trước
try:
    from backend.llm.providers.gemini.gemini_client import GeminiClient
    from backend.llm.providers.gemini.gemini_config import GeminiConfig
    from qdrant_client import QdrantClient
    from backend.db.neo4j_client import Neo4jClient
    from backend.db.vector_db import VectorDBClient
    from backend.retrieval.dual_level_retriever import (
        retrieve_from_knowledge_graph,
        retrieve_level1_nodes,
        retrieve_level2_references,
        format_retrieval_results)
except ImportError as e:
    print(f"❌ Lỗi import modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
# Import trực tiếp các modules cần thiết
try:
    from backend.db.neo4j_client import Neo4jClient
    from backend.db.vector_db import VectorDBClient
    from backend.utils.logging import get_logger
    from backend.pipeline.prompts.pipeline_prompts import PROMPTS
    print("✅ Đã import db modules, utils.logging và pipeline.prompts từ backend")
except ImportError as e:
    print(f"⚠️  Không thể import một số modules: {e}")
    print("⚠️  Một số chức năng có thể không hoạt động")


async def initialize_clients():
    """Khởi tạo các clients cần thiết"""
    print("=" * 60)
    print("🔧 KHỞI TẠO CLIENTS")
    print("=" * 60)

    clients = {}

    # 1. Gemini Client
    print("\n📊 Khởi tạo Gemini Client...")
    try:
        gemini_config = GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY_2") or os.getenv(
                "GEMINI_API_KEY_1"),
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        clients["gemini_client"] = GeminiClient(gemini_config)
        print("✅ Gemini Client đã được khởi tạo")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Gemini Client: {e}")
        return None

    # 2. Tạo Gemini Embedding Wrapper (thay thế Ollama)
    print("\n📊 Tạo Gemini Embedding Wrapper...")
    try:
        # Tạo một wrapper class để sử dụng Gemini cho embeddings
        class GeminiEmbeddingWrapper:
            def __init__(self, gemini_client, target_dimension=384):
                self.gemini_client = gemini_client
                # Luôn sử dụng 384 chiều cho embeddings (theo Qdrant collection)
                self.target_dimension = target_dimension

            def _resize_embedding(self, embedding, target_dim):
                """Resize embedding về target dimension bằng cách truncate hoặc pad"""
                if len(embedding) == target_dim:
                    return embedding
                elif len(embedding) > target_dim:
                    # Truncate nếu embedding lớn hơn (768 -> 384)
                    return embedding[:target_dim]
                else:
                    # Pad với zeros nếu embedding nhỏ hơn
                    return embedding + [0.0] * (target_dim - len(embedding))

            async def embed(self, texts):
                """Generate embeddings using Gemini và resize về target dimension (384)"""
                if isinstance(texts, str):
                    texts = [texts]

                if not texts:
                    return []

                try:
                    # Sử dụng Gemini để tạo embeddings (sync method từ gemini_client.py)
                    # gemini_client.embed() trả về List[List[float]]
                    # Wrap trong async để tương thích với interface
                    import asyncio
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.gemini_client.embed(texts)
                    )

                    # Kiểm tra và resize embeddings về target dimension
                    if embeddings and len(embeddings) > 0:
                        first_dim = len(embeddings[0]) if embeddings[0] else 0

                        if first_dim == 0:
                            print("⚠️  Embeddings rỗng, tạo embeddings giả")
                            return [[0.0] * self.target_dimension for _ in texts]

                        if first_dim != self.target_dimension:
                            print(
                                f"📊 Resizing embeddings từ {first_dim} chiều về {self.target_dimension} chiều")
                            embeddings = [self._resize_embedding(emb, self.target_dimension)
                                          for emb in embeddings]
                        else:
                            print(
                                f"✅ Embeddings đã có đúng {self.target_dimension} chiều")
                    else:
                        print(
                            "⚠️  Không nhận được embeddings từ Gemini, tạo embeddings giả")
                        embeddings = [
                            [0.0] * self.target_dimension for _ in texts]

                    return embeddings
                except Exception as e:
                    print(f"⚠️  Lỗi khi tạo embeddings: {e}")
                    import traceback
                    traceback.print_exc()
                    # Tạo embeddings giả với target dimension
                    return [[0.0] * self.target_dimension for _ in texts]

        # Tạo wrapper với target_dimension mặc định là 384
        # Sẽ được cập nhật sau nếu collection có dimension khác
        clients["ollama_client"] = GeminiEmbeddingWrapper(
            clients["gemini_client"], target_dimension=384)
        print("✅ Gemini Embedding Wrapper đã được tạo (thay thế Ollama)")
        print("   - Mặc định sử dụng 384 chiều cho embeddings")
    except Exception as e:
        print(f"❌ Lỗi tạo Gemini Embedding Wrapper: {e}")
        return None

    # 3. Qdrant Client (sử dụng trực tiếp QdrantClient vì dual_level_retriever dùng query_points)
    print("\n📊 Khởi tạo Qdrant Client...")
    try:
        qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        # Test connection
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections] if hasattr(
            collections, 'collections') else []
        print(f"✅ Qdrant Client đã được khởi tạo")
        print(f"   - Collections có sẵn: {collection_names}")

        # Kiểm tra collection "kg_level1"
        collection_dimension = None
        if "kg_level1" not in collection_names:
            print("⚠️  Collection 'kg_level1' chưa tồn tại trong Qdrant")
            print("💡 Cần tạo collection và import dữ liệu vào Qdrant")
        else:
            # Kiểm tra số lượng points và vector dimension trong collection
            try:
                collection_info = qdrant_client.get_collection("kg_level1")
                # Sử dụng points_count thay vì vectors_count
                points_count = 0
                if hasattr(collection_info, 'points_count'):
                    points_count = collection_info.points_count or 0
                elif hasattr(collection_info, 'vectors_count'):
                    points_count = collection_info.vectors_count or 0
                print(
                    f"   - Collection 'kg_level1' có {points_count} points")

                # Lấy vector dimension từ collection config
                try:
                    # Thử nhiều cách để lấy vector dimension
                    if hasattr(collection_info, 'config'):
                        config = collection_info.config
                        if hasattr(config, 'params'):
                            params = config.params
                            # Cách 1: Direct vectors config
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
                            # Cách 2: Check trong config trực tiếp
                            if not collection_dimension and hasattr(params, 'vector_size'):
                                collection_dimension = params.vector_size

                    # Nếu vẫn chưa có, thử lấy từ collection_info trực tiếp
                    if not collection_dimension:
                        # Thử các attribute khác
                        if hasattr(collection_info, 'vector_size'):
                            collection_dimension = collection_info.vector_size
                        elif hasattr(collection_info, 'config') and hasattr(collection_info.config, 'vector_size'):
                            collection_dimension = collection_info.config.vector_size

                    if collection_dimension:
                        print(f"   - Vector dimension: {collection_dimension}")
                    else:
                        print(
                            "   ⚠️  Không thể xác định vector dimension, sẽ dùng mặc định 384")
                        collection_dimension = 384  # Mặc định theo lỗi
                except Exception as dim_e:
                    print(f"   ⚠️  Lỗi khi lấy vector dimension: {dim_e}")
                    collection_dimension = 384  # Mặc định theo lỗi
            except Exception as e:
                print(f"   ⚠️  Không thể lấy thông tin collection: {e}")

        # Cập nhật target_dimension cho GeminiEmbeddingWrapper nếu collection có dimension khác 384
        # Nếu không có collection_dimension, giữ nguyên 384 (mặc định)
        if collection_dimension and collection_dimension != 384:
            if clients.get("ollama_client"):
                clients["ollama_client"].target_dimension = collection_dimension
                print(
                    f"   ✅ Đã cập nhật embedding target dimension: {collection_dimension}")
        elif collection_dimension == 384:
            print("   ✅ Collection dimension (384) khớp với mặc định")
        else:
            print("   ℹ️  Sử dụng vector dimension mặc định: 384")

        clients["qdrant_client"] = qdrant_client
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Qdrant Client: {e}")
        print("⚠️  Đảm bảo Qdrant đang chạy (docker compose up -d)")
        return None

    # 4. Neo4j Client
    print("\n📊 Khởi tạo Neo4j Client...")
    if Neo4jClient is None:
        print("⚠️  Neo4jClient không khả dụng")
        print("⚠️  Bỏ qua Neo4j tests")
        clients["neo4j_client"] = None
    else:
        try:
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER_NAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "123456789")

            # Khởi tạo Neo4jClient giống như trong check_retrival_qdrant2neo4j.py
            neo4j_client = Neo4jClient(
                uri=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password
            )
            print(f"✅ Đã tạo Neo4jClient cho {neo4j_uri}")

            # Verify connectivity trước khi test
            try:
                connected = await neo4j_client.verify_connectivity()
                if connected:
                    print("✅ Neo4j connectivity verified")
                else:
                    print("⚠️  Neo4j connectivity verification failed")
            except Exception as verify_e:
                print(f"⚠️  Lỗi khi verify connectivity: {verify_e}")

            # Test connection bằng cách thử một query đơn giản
            try:
                test_result = await neo4j_client.execute_query("RETURN 1 as test")
                if test_result:
                    print("✅ Neo4j connection test thành công")

                # Kiểm tra số lượng Level1 nodes trong database
                count_query = "MATCH (n:Level1) RETURN count(n) as total"
                count_result = await neo4j_client.execute_query(count_query)
                total_nodes = count_result[0].get(
                    'total', 0) if count_result and len(count_result) > 0 else 0
                print(f"   - Số Level1 nodes trong database: {total_nodes}")

                if total_nodes == 0:
                    print(
                        "   ⚠️  Database rỗng! Có thể dữ liệu nằm trong database khác.")
                    # Thử query với database khác
                    print("   💡 Kiểm tra xem có phải vấn đề về database name không...")
                else:
                    print(f"   ✅ Database có {total_nodes} Level1 nodes")
            except Exception as test_e:
                print(f"⚠️  Neo4j connection test failed: {test_e}")
                print("⚠️  Vẫn tiếp tục nhưng có thể có lỗi khi sử dụng Neo4j")
                import traceback
                traceback.print_exc()

            clients["neo4j_client"] = neo4j_client
            print("✅ Neo4j Client đã được khởi tạo")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Neo4j Client: {e}")
            print("⚠️  Đảm bảo Neo4j đang chạy và credentials đúng")
            clients["neo4j_client"] = None

    return clients


async def test_retrieve_from_knowledge_graph(clients):
    """Test hàm retrieve_from_knowledge_graph"""
    print("\n" + "=" * 60)
    print("🧪 TEST: retrieve_from_knowledge_graph")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("⚠️  Neo4j client không khả dụng, bỏ qua test này")
        return None

    if not clients.get("qdrant_client"):
        print("⚠️  Qdrant client không khả dụng, bỏ qua test này")
        return None

    # Test query và keywords
    test_query = "What are the symptoms and treatment of diabetes?"
    high_keywords = ["diabetes", "symptoms", "treatment"]
    low_keywords = ["blood sugar", "insulin", "glucose", "cathepsin L"]

    print(f"\n📝 Query: {test_query}")
    print(f"📝 High-level keywords: {high_keywords}")
    print(f"📝 Low-level keywords: {low_keywords}")

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

        print("\n✅ Kết quả retrieval:")
        l1_nodes = len(result.get('level1_nodes', []))
        l2_nodes = len(result.get('level2_nodes', []))
        relationships = len(result.get('relationships', []))

        print(f"   - Level 1 nodes: {l1_nodes}")
        print(f"   - Level 2 nodes: {l2_nodes}")
        print(f"   - Relationships: {relationships}")

        # Cảnh báo nếu không có kết quả
        if l1_nodes == 0:
            print("\n⚠️  KHÔNG TÌM THẤY NODES - Có thể do:")
            print("   1. Qdrant collection 'kg_level1' chưa có dữ liệu")
            print("   2. Neo4j database chưa có dữ liệu tương ứng")
            print("   3. Embeddings không match với data trong Qdrant")
            print("   4. Similarity threshold quá cao (hiện tại: 0.7)")
            print("   5. Entity IDs trong Qdrant payload không match với Neo4j")
            print("\n💡 Để test đầy đủ, bạn cần:")
            print("   - Import dữ liệu vào Qdrant (collection 'kg_level1')")
            print("   - Import dữ liệu tương ứng vào Neo4j")
            print(
                "   - Đảm bảo 'node_id' trong Qdrant payload match với 'id' trong Neo4j")
            print(
                "   - Đảm bảo embeddings trong Qdrant được tạo từ cùng model với query embeddings")

        # Hiển thị một số nodes mẫu
        if result.get('level1_nodes'):
            print("\n📋 Sample Level 1 nodes:")
            for i, node in enumerate(result['level1_nodes'][:3], 1):
                print(
                    f"   {i}. {node.get('name', 'Unknown')} ({node.get('entity_type', 'Unknown')}) - Score: {node.get('similarity_score', 0):.4f}")

        if result.get('level2_nodes'):
            print("\n📋 Sample Level 2 nodes:")
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
        print(f"\n❌ Lỗi khi test retrieve_from_knowledge_graph: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_retrieve_level1_nodes(clients):
    """Test hàm retrieve_level1_nodes"""
    print("\n" + "=" * 60)
    print("🧪 TEST: retrieve_level1_nodes")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("⚠️  Neo4j client không khả dụng, bỏ qua test này")
        return None

    if not clients.get("qdrant_client"):
        print("⚠️  Qdrant client không khả dụng, bỏ qua test này")
        return None

    # Tạo test embeddings
    test_keywords = ["diabetes", "treatment"]
    print(f"\n📝 Test keywords: {test_keywords}")

    try:
        # Generate embeddings using Gemini wrapper
        embeddings = await clients["ollama_client"].embed(test_keywords)
        print(f"✅ Đã tạo {len(embeddings)} embeddings bằng Gemini")

        # Retrieve Level 1 nodes
        # Lưu ý: retrieve_level1_nodes chỉ nhận top_k và similarity_threshold
        nodes = await retrieve_level1_nodes(
            embeddings=embeddings,
            qdrant_client=clients["qdrant_client"],
            neo4j_client=clients["neo4j_client"],
            top_k=5,
            similarity_threshold=0.7
        )

        num_nodes = len(nodes)
        print(f"\n✅ Đã retrieve {num_nodes} Level 1 nodes")

        if num_nodes == 0:
            print("\n⚠️  KHÔNG TÌM THẤY NODES - Có thể do:")
            print("   1. Qdrant collection 'kg_level1' chưa có dữ liệu")
            print("   2. Neo4j database chưa có dữ liệu tương ứng")
            print("   3. Embeddings không match với data trong Qdrant")
            print("   4. Similarity threshold quá cao (hiện tại: 0.7)")
            print(
                "   5. Entity IDs trong Qdrant payload không match với Neo4j 'id' field")
        elif nodes:
            print("\n📋 Sample nodes:")
            for i, node in enumerate(nodes[:3], 1):
                print(
                    f"   {i}. {node.get('name', 'Unknown')} (score: {node.get('similarity_score', 0):.4f})")

        return nodes

    except Exception as e:
        print(f"\n❌ Lỗi khi test retrieve_level1_nodes: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_retrieve_level2_references(clients, level1_nodes):
    """Test hàm retrieve_level2_references"""
    print("\n" + "=" * 60)
    print("🧪 TEST: retrieve_level2_references")
    print("=" * 60)

    if not clients.get("neo4j_client"):
        print("⚠️  Neo4j client không khả dụng, bỏ qua test này")
        return None, None

    if not level1_nodes:
        print("⚠️  Không có Level 1 nodes để test, bỏ qua test này")
        return None, None

    try:
        # Lấy một vài Level 1 nodes để test
        test_level1_nodes = level1_nodes[:3] if len(
            level1_nodes) >= 3 else level1_nodes
        print(f"\n📝 Test với {len(test_level1_nodes)} Level 1 nodes")

        level2_nodes, relationships = await retrieve_level2_references(
            level1_nodes=test_level1_nodes,
            neo4j_client=clients["neo4j_client"],
            max_references=5
        )

        print(f"\n✅ Đã retrieve:")
        print(f"   - Level 2 nodes: {len(level2_nodes)}")
        print(f"   - Relationships: {len(relationships)}")

        if level2_nodes:
            print("\n📋 Sample Level 2 nodes:")
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
        print(f"\n❌ Lỗi khi test retrieve_level2_references: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_format_functions(level1_nodes, level2_nodes, relationships):
    """Test các hàm format"""
    print("\n" + "=" * 60)
    print("🧪 TEST: Format Functions")
    print("=" * 60)

    if not level1_nodes:
        print("⚠️  Không có nodes để test, bỏ qua test này")
        return

    try:
        # Test format_retrieval_results
        print("\n📊 Test format_retrieval_results...")
        formatted_text = format_retrieval_results(
            level1_nodes=level1_nodes[:5] if len(
                level1_nodes) >= 5 else level1_nodes,
            level2_nodes=level2_nodes[:5] if level2_nodes and len(
                level2_nodes) >= 5 else (level2_nodes or [])[:5],
            relationships=relationships[:5] if relationships and len(
                relationships) >= 5 else (relationships or [])[:5]
        )
        print(f"✅ Formatted text length: {len(formatted_text)} characters")
        preview = formatted_text[:500] if len(
            formatted_text) > 500 else formatted_text
        print(f"📄 Preview (first 500 chars):\n{preview}...")

        # Hiển thị thêm thông tin về format
        print("\n📊 Thông tin về formatted text:")
        print(f"   - Tổng số ký tự: {len(formatted_text)}")
        print(f"   - Số dòng: {len(formatted_text.split(chr(10)))}")

        # Kiểm tra các section chính
        if "# KEY CONCEPTS" in formatted_text:
            print("   ✅ Có section KEY CONCEPTS")
        if "# RELATIONSHIPS" in formatted_text:
            print("   ✅ Có section RELATIONSHIPS")
        if "# DETAILED INFORMATION" in formatted_text:
            print("   ✅ Có section DETAILED INFORMATION")

    except Exception as e:
        print(f"\n❌ Lỗi khi test format functions: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Hàm main để chạy tất cả các tests"""
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU TEST DUAL LEVEL RETRIEVER")
    print("=" * 60)

    # Khởi tạo clients
    clients = await initialize_clients()
    if not clients:
        print("\n❌ Không thể khởi tạo clients. Vui lòng kiểm tra cấu hình.")
        return

    # Test 1: retrieve_from_knowledge_graph
    kg_result = await test_retrieve_from_knowledge_graph(clients)

    # Test 2: retrieve_level1_nodes (nếu test 1 thành công)
    level1_nodes = None
    if kg_result:
        level1_nodes = kg_result.get('level1_nodes', [])

    # Nếu không có nodes từ test 1, test riêng retrieve_level1_nodes
    if not level1_nodes:
        level1_nodes = await test_retrieve_level1_nodes(clients)

    # Test 3: retrieve_level2_references
    level2_nodes, relationships = await test_retrieve_level2_references(clients, level1_nodes)

    # Test 4: Format functions
    await test_format_functions(level1_nodes, level2_nodes, relationships)

    # Đóng connections
    print("\n" + "=" * 60)
    print("🔌 ĐÓNG CONNECTIONS")
    print("=" * 60)
    try:
        if clients.get("neo4j_client"):
            # Kiểm tra xem có method close không
            if hasattr(clients["neo4j_client"], "close"):
                await clients["neo4j_client"].close()
                print("✅ Đã đóng Neo4j connection")
            else:
                print("ℹ️  Neo4j client không có method close()")
    except Exception as e:
        print(f"⚠️  Lỗi khi đóng Neo4j connection: {e}")

    print("\n" + "=" * 60)
    print("✅ TẤT CẢ CÁC TEST ĐÃ HOÀN THÀNH!")
    print("=" * 60)

    # Tóm tắt kết quả
    print("\n📊 TÓM TẮT:")
    if kg_result:
        total_nodes = (len(kg_result.get('level1_nodes', [])) +
                       len(kg_result.get('level2_nodes', [])))
        if total_nodes == 0:
            print("⚠️  Không có nodes nào được retrieve")
            print("💡 Để test đầy đủ, cần:")
            print("   - Import dữ liệu vào Qdrant (collection 'kg_level1')")
            print("   - Import dữ liệu tương ứng vào Neo4j")
            print(
                "   - Đảm bảo 'node_id' trong Qdrant match với 'id' trong Neo4j")
        else:
            print(f"✅ Đã retrieve {total_nodes} nodes từ knowledge graph")
            print(f"   - Level 1: {len(kg_result.get('level1_nodes', []))}")
            print(f"   - Level 2: {len(kg_result.get('level2_nodes', []))}")
    else:
        print("⚠️  Không thể test retrieval - có thể do Qdrant hoặc Neo4j chưa có dữ liệu")


if __name__ == "__main__":
    # Chạy async main
    asyncio.run(main())
