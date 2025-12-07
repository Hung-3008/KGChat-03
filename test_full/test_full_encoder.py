"""
Test script hoàn chỉnh cho toàn bộ pipeline xử lý câu hỏi
Tích hợp:
1. Query Intent Classification và Keyword Extraction (từ test_query_intent_extractkey.py)
2. Triple Level Retriever - truy vấn đồ thị (chỉ tầng 1 và tầng 2, từ test_triple_lv_source_v1.py)
"""
import sys
import os
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Thêm cả thư mục gốc và backend vào path
root_path = Path(__file__).parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(backend_path))

# Import các modules cần thiết
try:
    from backend.llm.providers.gemini.gemini_client import GeminiClient
    from backend.llm.providers.gemini.gemini_config import GeminiConfig
    from backend.pipeline.query_analyzer import QueryIntent
    from backend.pipeline.kg_pipeline import (
        process_query_full_pipeline
    )
    from backend.encoders.transformer_encoder import TransformerEncoder
    from qdrant_client import QdrantClient
    from backend.db.neo4j_client import Neo4jClient
    print("✅ Đã import các modules thành công")
except ImportError as e:
    print(f"❌ Lỗi import modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


async def initialize_clients():
    """Khởi tạo tất cả các clients cần thiết"""
    print("=" * 80)
    print("🔧 KHỞI TẠO CLIENTS")
    print("=" * 80)

    clients = {}

    # 1. Gemini Client
    print("\n📌 Khởi tạo Gemini Client...")
    try:
        gemini_config = GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY_2") or os.getenv(
                "GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY_16"),
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        clients["gemini_client"] = GeminiClient(gemini_config)
        print("✅ Gemini Client đã được khởi tạo")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Gemini Client: {e}")
        return None

    # 2. Khởi tạo Transformer Encoder
    print("\n📌 Khởi tạo Transformer Encoder...")
    try:
        transformer_encoder = TransformerEncoder(
            model_name="NeuML/pubmedbert-base-embeddings",
            device="cpu"
        )

        # Sử dụng trực tiếp TransformerEncoder
        clients["ollama_client"] = transformer_encoder
        print("✅ Transformer Encoder đã được khởi tạo")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Transformer Encoder: {e}")
        traceback.print_exc()
        return None

    # 3. Qdrant Client
    print("\n📌 Khởi tạo Qdrant Client...")
    try:
        qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        # Kiểm tra kết nối
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections] if hasattr(
            collections, 'collections') else []
        print("✅ Qdrant Client đã được khởi tạo")
        print(f"   - Collections có sẵn: {collection_names}")

        # Kiểm tra collection "kg_lv1_nodes" và lấy dimension
        collection_dimension = None
        if "kg_lv1_nodes" not in collection_names:
            print("⚠️  Cảnh báo: Collection 'kg_lv1_nodes' không tồn tại")
        else:
            try:
                collection_info = qdrant_client.get_collection("kg_lv1_nodes")
                points_count = 0
                if hasattr(collection_info, 'points_count'):
                    points_count = collection_info.points_count or 0
                elif hasattr(collection_info, 'vectors_count'):
                    points_count = collection_info.vectors_count or 0
                print(
                    f"   - Collection 'kg_lv1_nodes' có {points_count} points")

                # Lấy vector dimension
                if hasattr(collection_info, 'config'):
                    config = collection_info.config
                    if hasattr(config, 'params'):
                        params = config.params
                        if hasattr(params, 'vectors'):
                            vectors = params.vectors
                            if hasattr(vectors, 'size'):
                                collection_dimension = vectors.size
                            elif isinstance(vectors, dict):
                                for vec_name, vec_config in vectors.items():
                                    if hasattr(vec_config, 'size'):
                                        collection_dimension = vec_config.size
                                        break

                if collection_dimension:
                    print(f"   - Vector dimension: {collection_dimension}")
            except Exception as e:
                print(f"   ⚠️  Lỗi lấy thông tin collection: {e}")

        clients["qdrant_client"] = qdrant_client
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Qdrant Client: {e}")
        print("⚠️  Đảm bảo Qdrant đang chạy (docker compose up -d)")
        return None

    # 4. Neo4j Client
    print("\n📌 Khởi tạo Neo4j Client...")
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER_NAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "123456789")

        neo4j_client = Neo4jClient(
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        print(f"✅ Đã tạo Neo4jClient cho {neo4j_uri}")

        # Kiểm tra kết nối
        try:
            connected = await neo4j_client.verify_connectivity()
            if connected:
                print("✅ Đã xác minh kết nối Neo4j")

            # Kiểm tra số lượng Level1 nodes
            count_query = "MATCH (n:Level1) RETURN count(n) as total"
            count_result = await neo4j_client.execute_query(count_query)
            total_nodes = count_result[0].get(
                'total', 0) if count_result and len(count_result) > 0 else 0
            print(f"   - Số lượng Level1 nodes trong database: {total_nodes}")

            if total_nodes == 0:
                print("   ⚠️  Cảnh báo: Database trống!")
        except Exception as test_e:
            print(f"⚠️  Cảnh báo: Lỗi kiểm tra kết nối Neo4j: {test_e}")

        clients["neo4j_client"] = neo4j_client
        print("✅ Neo4j Client đã được khởi tạo")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Neo4j Client: {e}")
        print("⚠️  Đảm bảo Neo4j đang chạy và credentials đúng")
        clients["neo4j_client"] = None

    return clients


async def main():
    """Hàm main để chạy test"""
    print("\n" + "=" * 80)
    print("🧪 TEST TOÀN BỘ PIPELINE - QUERY INTENT + KEYWORD EXTRACTION + GRAPH RETRIEVAL + LLM ANSWER")
    print("=" * 80)

    # Khởi tạo clients
    clients = await initialize_clients()
    if not clients:
        print("\n❌ Không thể khởi tạo clients. Dừng test.")
        return

    # Danh sách các query test
    test_queries = [
        # Healthcare queries
        "What are the xerostomia, cathepsin L and squamous cell carcinoma?",
    ]

    # Chạy test cho từng query
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*80}")

        result = await process_query_full_pipeline(query, clients=clients)
        if result:
            results.append(result)

        # Nghỉ một chút giữa các test để tránh rate limit
        if i < len(test_queries):
            await asyncio.sleep(2)

    # Tổng kết cuối cùng
    print("\n\n" + "=" * 80)
    print("📈 TỔNG KẾT TẤT CẢ CÁC TEST")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Query: {result['query']}")
        if result['intent']:
            intent_name = result['intent'].name
            print(f"   Intent: {intent_name}")
            if result['intent'] == QueryIntent.HEALTHCARE_RELATED:
                print(
                    f"   High-level keywords: {len(result['high_level_keywords'])}")
                print(
                    f"   Low-level keywords: {len(result['low_level_keywords'])}")
                if result['retrieval_result']:
                    print(
                        f"   Level 1 nodes: {len(result['retrieval_result'].get('level1_nodes', []))}")
                    print(
                        f"   Level 2 nodes: {len(result['retrieval_result'].get('level2_nodes', []))}")
                    print(
                        f"   Relationships: {len(result['retrieval_result'].get('relationships', []))}")
            else:
                print(
                    "   Keywords: Không trích xuất (chỉ trích xuất cho HEALTHCARE_RELATED)")
            if result.get('saved_file'):
                print(f"   📁 Prompt file: {result['saved_file']}")
            if result.get('answer_file'):
                print(f"   📄 Answer file: {result['answer_file']}")
            if result.get('llm_answer'):
                answer_preview = result['llm_answer'][:200] if len(
                    result['llm_answer']) > 200 else result['llm_answer']
                print(f"   💬 Answer preview: {answer_preview}...")
    print("=" * 80)

    # Đóng kết nối
    print("\n" + "=" * 80)
    print("🔒 ĐÓNG KẾT NỐI")
    print("=" * 80)
    try:
        if clients.get("neo4j_client"):
            if hasattr(clients["neo4j_client"], "close"):
                await clients["neo4j_client"].close()
                print("✅ Đã đóng kết nối Neo4j")
    except Exception as e:
        print(f"⚠️  Lỗi đóng kết nối: {e}")

    print("\n✅ HOÀN TẤT TẤT CẢ CÁC TEST")


if __name__ == "__main__":
    asyncio.run(main())
