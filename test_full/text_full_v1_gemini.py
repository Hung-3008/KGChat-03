"""
Test script hoàn chỉnh cho toàn bộ pipeline xử lý câu hỏi
Tích hợp:
1. Query Intent Classification và Keyword Extraction (từ test_query_intent_extractkey.py)
2. Triple Level Retriever - truy vấn đồ thị (chỉ tầng 1 và tầng 2, từ test_triple_lv_source_v1.py)
"""
import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, Any
import re

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
    from backend.pipeline.query_analyzer import analyze_query, QueryIntent
    from backend.pipeline.keyword_extractor import extract_keywords
    from backend.retrieval.triple_level_retriever import (
        retrieve_from_knowledge_graph,
        format_retrieval_results
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

    # 2. Tạo Transformer Encoder Wrapper (thay thế Gemini/Ollama embedding)
    print("\n📌 Khởi tạo Transformer Encoder...")
    try:
        class TransformerEmbeddingWrapper:
            def __init__(self, transformer_encoder, target_dimension=768):
                self.transformer_encoder = transformer_encoder
                self.target_dimension = target_dimension

            def _resize_embedding(self, embedding, target_dim):
                """Resize embedding về target dimension"""
                if len(embedding) == target_dim:
                    return embedding
                elif len(embedding) > target_dim:
                    return embedding[:target_dim]
                else:
                    return embedding + [0.0] * (target_dim - len(embedding))

            async def embed(self, texts):
                """Tạo embeddings sử dụng Transformer Encoder"""
                if isinstance(texts, str):
                    texts = [texts]

                if not texts:
                    return []

                try:
                    import asyncio
                    import numpy as np

                    # Chạy embedding trong executor để không block event loop
                    loop = asyncio.get_event_loop()
                    encoder = self.transformer_encoder
                    embeddings_tensor = await loop.run_in_executor(
                        None,
                        lambda: encoder.embed_to_numpy(texts)
                    )

                    # Convert numpy array sang list of lists
                    if isinstance(embeddings_tensor, np.ndarray):
                        embeddings = embeddings_tensor.tolist()
                    else:
                        # Nếu là torch.Tensor, convert sang numpy rồi sang list
                        if hasattr(embeddings_tensor, 'numpy'):
                            embeddings = embeddings_tensor.numpy().tolist()
                        else:
                            embeddings = list(embeddings_tensor)

                    if embeddings and len(embeddings) > 0:
                        first_dim = len(embeddings[0]) if embeddings[0] else 0

                        if first_dim == 0:
                            print(
                                "⚠️  Cảnh báo: Embeddings rỗng, tạo dummy embeddings")
                            return [[0.0] * self.target_dimension for _ in texts]

                        if first_dim != self.target_dimension:
                            print(
                                f"🔄 Resize embeddings từ {first_dim} về {self.target_dimension} dimensions")
                            embeddings = [self._resize_embedding(emb, self.target_dimension)
                                          for emb in embeddings]
                        else:
                            print(
                                f"✅ Embeddings đã có đúng {self.target_dimension} dimensions")
                    else:
                        print(
                            "⚠️  Cảnh báo: Không nhận được embeddings từ Transformer Encoder")
                        embeddings = [
                            [0.0] * self.target_dimension for _ in texts]

                    return embeddings
                except Exception as e:
                    print(f"❌ Lỗi tạo embeddings: {e}")
                    import traceback
                    traceback.print_exc()
                    return [[0.0] * self.target_dimension for _ in texts]

        # Khởi tạo Transformer Encoder
        transformer_encoder = TransformerEncoder(
            model_name="NeuML/pubmedbert-base-embeddings",
            device="cpu"
        )

        # Tạo wrapper với dimension mặc định 768
        clients["ollama_client"] = TransformerEmbeddingWrapper(
            transformer_encoder, target_dimension=768)
        print("✅ Transformer Encoder Wrapper đã được tạo")
    except Exception as e:
        print(f"❌ Lỗi tạo Transformer Encoder Wrapper: {e}")
        import traceback
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
        print(f"✅ Qdrant Client đã được khởi tạo")
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
                    # Cập nhật target_dimension cho wrapper
                    if clients.get("ollama_client"):
                        clients["ollama_client"].target_dimension = collection_dimension
                        print(
                            f"   - Đã cập nhật embedding dimension: {collection_dimension}")
                else:
                    print("   - Sử dụng dimension mặc định: 768")
                    collection_dimension = 768
                    # Cập nhật target_dimension cho wrapper
                    if clients.get("ollama_client"):
                        clients["ollama_client"].target_dimension = collection_dimension
            except Exception as e:
                print(f"   ⚠️  Lỗi lấy thông tin collection: {e}")
                collection_dimension = 768

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


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Chuyển đổi text thành tên file hợp lệ"""
    # Loại bỏ các ký tự đặc biệt
    text = re.sub(r'[^\w\s-]', '', text)
    # Thay thế khoảng trắng bằng dấu gạch dưới
    text = re.sub(r'[-\s]+', '_', text)
    # Giới hạn độ dài
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip('_')


async def generate_answer_with_llm(
    query: str,
    formatted_text: str,
    gemini_client: Any,
    conversation_history: Optional[str] = None
) -> Optional[str]:
    """
    Tạo câu trả lời từ LLM Gemini sử dụng retrieval context

    Args:
        query: Câu hỏi của người dùng
        formatted_text: Text đã được format từ format_retrieval_results
        gemini_client: Gemini client để gọi LLM (GeminiClient instance)
        conversation_history: Lịch sử hội thoại (tùy chọn)

    Returns:
        Câu trả lời từ LLM hoặc None nếu lỗi
    """
    try:
        # Tạo prompt RAG
        rag_prompt = create_rag_prompt(
            query, formatted_text, conversation_history)

        # System prompt cho LLM
        system_prompt = (
            "You are a helpful assistant that provides accurate and detailed answers "
            "based on the provided knowledge base. Use only the information from the "
            "Knowledge Base section to answer the user's query. If the information "
            "is not available in the Knowledge Base, clearly state that you don't have "
            "enough information to answer the question."
        )

        # Gọi Gemini để tạo câu trả lời
        # Vì generate() là sync method, chạy trong executor để không block event loop
        print("\n🤖 Đang gọi Gemini LLM để tạo câu trả lời...")
        print("   ⏳ Đang xử lý (có thể mất vài giây)...")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.generate(
                user_prompt=rag_prompt,
                system_prompt=system_prompt
            )
        )

        # Extract message từ LLMResponse object
        # GeminiClient.generate() trả về LLMResponse với attribute 'message'
        if hasattr(response, 'message'):
            answer = str(response.message).strip()
        elif hasattr(response, 'text'):  # Fallback nếu có attribute 'text'
            answer = str(response.text).strip()
        elif isinstance(response, dict):
            answer = response.get("message", "").strip()
            if isinstance(answer, dict):
                answer = answer.get("content", "").strip()
        else:
            answer = str(response).strip()

        if not answer:
            print("⚠️  LLM trả về câu trả lời rỗng")
            return None

        print("✅ Đã nhận được câu trả lời từ LLM")
        return answer

    except Exception as e:
        print(f"❌ Lỗi khi gọi LLM: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_rag_prompt(query: str, formatted_text: str, conversation_history: Optional[str] = None) -> str:
    """
    Tạo prompt RAG sẵn sàng để đưa vào LLM

    Args:
        query: Câu hỏi của người dùng
        formatted_text: Text đã được format từ format_retrieval_results
        conversation_history: Lịch sử hội thoại (tùy chọn)

    Returns:
        Prompt string sẵn sàng để đưa vào LLM
    """
    prompt_parts = []

    # Role và Goal
    prompt_parts.append("---Role---")
    prompt_parts.append("")
    prompt_parts.append(
        "You are a helpful assistant responding to user query about Knowledge Base provided below.")
    prompt_parts.append("")
    prompt_parts.append("---Goal---")
    prompt_parts.append("")
    prompt_parts.append(
        "Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.")
    prompt_parts.append("")

    # Conversation History
    prompt_parts.append("---Conversation History---")
    if conversation_history:
        prompt_parts.append(conversation_history)
    else:
        prompt_parts.append("(No previous conversation)")
    prompt_parts.append("")

    # User Query
    prompt_parts.append("---User Query---")
    prompt_parts.append(query)
    prompt_parts.append("")

    # Knowledge Base
    prompt_parts.append("---Knowledge Base---")
    prompt_parts.append("")
    prompt_parts.append(formatted_text)
    prompt_parts.append("")

    # Response Rules
    prompt_parts.append("---Response Rules---")
    prompt_parts.append("")
    prompt_parts.append(
        "- Target format and length: comprehensive and detailed")
    prompt_parts.append(
        "- Use markdown formatting with appropriate section headings")
    prompt_parts.append(
        "- Please respond in the same language as the user's question")
    prompt_parts.append(
        "- Ensure the response maintains continuity with the conversation history")
    prompt_parts.append(
        "- List up to 5 most important reference sources at the end under \"References\" section")
    prompt_parts.append(
        "- If you don't know the answer, just say so")
    prompt_parts.append(
        "- Do not make anything up. Do not include information not provided by the Knowledge Base")

    return "\n".join(prompt_parts)


def save_retrieval_result(query: str, formatted_text: str, result: dict, conversation_history=None) -> Optional[str]:
    """
    Lưu kết quả retrieval đã format vào folder output_pipeline_retrie
    Format theo cấu trúc prompt RAG để có thể dùng trực tiếp cho LLM

    Args:
        query: Câu hỏi gốc
        formatted_text: Text đã được format từ format_retrieval_results
        result: Dictionary chứa toàn bộ kết quả
        conversation_history: Lịch sử hội thoại (tùy chọn)

    Returns:
        Đường dẫn file đã lưu hoặc None nếu lỗi
    """
    try:
        # Tạo folder output_pipeline_retrie nếu chưa có
        root_path = Path(__file__).parent.parent
        output_dir = root_path / "output_pipeline_retrie"
        output_dir.mkdir(exist_ok=True)

        # Tạo tên file dựa trên query và timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = sanitize_filename(query, max_length=50)
        filename = f"prompt_{timestamp}_{query_sanitized}.txt"
        file_path = output_dir / filename

        # Tạo prompt RAG sẵn sàng để đưa vào LLM
        rag_prompt = create_rag_prompt(
            query, formatted_text, conversation_history)

        # Tạo nội dung file với prompt và metadata
        content_parts = []

        # Phần prompt chính (sẵn sàng để copy vào LLM)
        content_parts.append("=" * 80)
        content_parts.append(
            "PROMPT READY FOR LLM (Copy phần dưới để sử dụng)")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(rag_prompt)
        content_parts.append("")

        # Metadata section (cho reference)
        content_parts.append("=" * 80)
        content_parts.append("METADATA (For reference only)")
        content_parts.append("=" * 80)
        content_parts.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_parts.append(
            f"Intent: {result.get('intent').name if result.get('intent') else 'N/A'}")

        # Keywords
        if result.get('intent') == QueryIntent.HEALTHCARE_RELATED:
            content_parts.append(
                f"High-level keywords: {', '.join(result.get('high_level_keywords', []))}")
            content_parts.append(
                f"Low-level keywords: {', '.join(result.get('low_level_keywords', []))}")

        # Retrieval statistics
        if result.get('retrieval_result'):
            retrieval = result['retrieval_result']
            content_parts.append(f"Retrieval Statistics:")
            content_parts.append(
                f"  - Level 1 nodes: {len(retrieval.get('level1_nodes', []))}")
            content_parts.append(
                f"  - Level 2 nodes: {len(retrieval.get('level2_nodes', []))}")
            content_parts.append(
                f"  - Relationships: {len(retrieval.get('relationships', []))}")

        # Ghi file
        full_content = "\n".join(content_parts)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        return str(file_path)

    except Exception as e:
        print(f"⚠️  Lỗi khi lưu file: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_llm_answer(query: str, answer: str, result: dict, prompt_file: Optional[str] = None) -> Optional[str]:
    """
    Lưu câu trả lời từ LLM vào file

    Args:
        query: Câu hỏi gốc
        answer: Câu trả lời từ LLM
        result: Dictionary chứa toàn bộ kết quả
        prompt_file: Đường dẫn file prompt đã lưu (tùy chọn)

    Returns:
        Đường dẫn file đã lưu hoặc None nếu lỗi
    """
    try:
        # Tạo folder output_pipeline_retrie nếu chưa có
        root_path = Path(__file__).parent.parent
        output_dir = root_path / "output_pipeline_retrie"
        output_dir.mkdir(exist_ok=True)

        # Tạo tên file dựa trên query và timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = sanitize_filename(query, max_length=50)
        filename = f"answer_{timestamp}_{query_sanitized}.txt"
        file_path = output_dir / filename

        # Tạo nội dung file
        content_parts = []

        content_parts.append("=" * 80)
        content_parts.append("LLM ANSWER - KNOWLEDGE GRAPH QUERY")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(f"Query: {query}")
        content_parts.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_parts.append(
            f"Intent: {result.get('intent').name if result.get('intent') else 'N/A'}")

        if prompt_file:
            content_parts.append(f"Prompt File: {prompt_file}")

        content_parts.append("")
        content_parts.append("=" * 80)
        content_parts.append("ANSWER")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(answer)
        content_parts.append("")

        # Metadata
        content_parts.append("=" * 80)
        content_parts.append("METADATA")
        content_parts.append("=" * 80)

        if result.get('intent') == QueryIntent.HEALTHCARE_RELATED:
            content_parts.append(
                f"High-level keywords: {', '.join(result.get('high_level_keywords', []))}")
            content_parts.append(
                f"Low-level keywords: {', '.join(result.get('low_level_keywords', []))}")

        if result.get('retrieval_result'):
            retrieval = result['retrieval_result']
            content_parts.append(f"Retrieval Statistics:")
            content_parts.append(
                f"  - Level 1 nodes: {len(retrieval.get('level1_nodes', []))}")
            content_parts.append(
                f"  - Level 2 nodes: {len(retrieval.get('level2_nodes', []))}")
            content_parts.append(
                f"  - Relationships: {len(retrieval.get('relationships', []))}")

        content_parts.append(f"Answer length: {len(answer)} characters")

        # Ghi file
        full_content = "\n".join(content_parts)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        return str(file_path)

    except Exception as e:
        print(f"⚠️  Lỗi khi lưu câu trả lời: {e}")
        import traceback
        traceback.print_exc()
        return None


async def process_query_full_pipeline(query: str, conversation_history=None, clients=None):
    """
    Xử lý câu hỏi qua toàn bộ pipeline:
    1. Phân loại intent
    2. Trích xuất keywords (nếu HEALTHCARE_RELATED)
    3. Truy vấn đồ thị (chỉ tầng 1 và tầng 2)
    4. Tạo câu trả lời từ LLM Gemini
    """
    print("\n" + "=" * 80)
    print("🚀 XỬ LÝ CÂU HỎI - TOÀN BỘ PIPELINE")
    print("=" * 80)
    print(f"📝 Query: {query}")
    print("=" * 80)

    if not clients:
        print("❌ Không có clients, không thể xử lý")
        return None

    gemini_client = clients.get("gemini_client")
    if not gemini_client:
        print("❌ Không có Gemini client")
        return None

    result = {
        "query": query,
        "intent": None,
        "high_level_keywords": [],
        "low_level_keywords": [],
        "retrieval_result": None,
        "formatted_text": "",
        "saved_file": None,
        "llm_answer": None,
        "answer_file": None
    }

    # BƯỚC 1: Phân loại intent
    print("\n" + "-" * 80)
    print("📋 BƯỚC 1: PHÂN LOẠI INTENT")
    print("-" * 80)
    try:
        intent = await analyze_query(
            query=query,
            conversation_history=conversation_history,
            client=gemini_client
        )
        result["intent"] = intent
        print(f"✅ Intent: {intent.name} ({intent.value})")
    except Exception as e:
        print(f"❌ Lỗi phân loại intent: {e}")
        import traceback
        traceback.print_exc()
        return result

    # BƯỚC 2: Trích xuất keywords (chỉ khi HEALTHCARE_RELATED)
    if intent == QueryIntent.HEALTHCARE_RELATED:
        print("\n" + "-" * 80)
        print("🔑 BƯỚC 2: TRÍCH XUẤT KEYWORDS")
        print("-" * 80)
        try:
            high_level_keywords, low_level_keywords = await extract_keywords(
                query=query,
                conversation_history=conversation_history,
                llm_client=gemini_client
            )
            result["high_level_keywords"] = high_level_keywords
            result["low_level_keywords"] = low_level_keywords

            print(f"✅ High-level keywords ({len(high_level_keywords)}):")
            for i, keyword in enumerate(high_level_keywords, 1):
                print(f"   {i}. {keyword}")

            print(f"\n✅ Low-level keywords ({len(low_level_keywords)}):")
            for i, keyword in enumerate(low_level_keywords, 1):
                print(f"   {i}. {keyword}")
        except Exception as e:
            print(f"❌ Lỗi trích xuất keywords: {e}")
            import traceback
            traceback.print_exc()
            return result

        # BƯỚC 3: Truy vấn đồ thị (chỉ tầng 1 và tầng 2)
        if high_level_keywords or low_level_keywords:
            print("\n" + "-" * 80)
            print("🔍 BƯỚC 3: TRUY VẤN ĐỒ THỊ (TẦNG 1 & TẦNG 2)")
            print("-" * 80)

            neo4j_client = clients.get("neo4j_client")
            ollama_client = clients.get("ollama_client")
            qdrant_client = clients.get("qdrant_client")

            if not neo4j_client or not ollama_client or not qdrant_client:
                print("❌ Thiếu clients cần thiết cho truy vấn đồ thị")
                print(f"   - Neo4j: {'✅' if neo4j_client else '❌'}")
                print(
                    f"   - Transformer Encoder (Embedding): {'✅' if ollama_client else '❌'}")
                print(f"   - Qdrant: {'✅' if qdrant_client else '❌'}")
                return result

            try:
                # Hiển thị keywords trước khi retrieval
                print(f"\n📌 Keywords để truy vấn:")
                print(f"   High-level: {high_level_keywords}")
                print(f"   Low-level: {low_level_keywords}")

                # Thử với similarity threshold thấp hơn nếu không tìm thấy
                similarity_threshold = 0.5  # Giảm từ 0.7 xuống 0.5 để tìm được nhiều nodes hơn
                print(
                    f"\n🔍 Đang truy vấn với similarity threshold: {similarity_threshold}")

                retrieval_result = await retrieve_from_knowledge_graph(
                    high_level_keywords=high_level_keywords,
                    low_level_keywords=low_level_keywords,
                    neo4j_client=neo4j_client,
                    ollama_client=ollama_client,
                    qdrant_client=qdrant_client,
                    top_k=10,  # Tăng top_k để tìm nhiều nodes hơn
                    similarity_threshold=similarity_threshold
                )

                result["retrieval_result"] = retrieval_result

                # Hiển thị kết quả
                l1_nodes = len(retrieval_result.get('level1_nodes', []))
                l2_nodes = len(retrieval_result.get('level2_nodes', []))
                relationships = len(retrieval_result.get('relationships', []))

                print(f"\n✅ Kết quả truy vấn:")
                print(f"   - Level 1 nodes: {l1_nodes}")
                print(f"   - Level 2 nodes: {l2_nodes}")
                print(f"   - Relationships: {relationships}")

                if l1_nodes == 0:
                    print("\n⚠️  Cảnh báo: KHÔNG TÌM THẤY NODES")
                    print("   Có thể do:")
                    print("   1. Qdrant collection 'kg_lv1_nodes' không có dữ liệu")
                    print("   2. Neo4j database không có dữ liệu tương ứng")
                    print("   3. Embeddings không khớp với dữ liệu trong Qdrant")
                    print(
                        f"   4. Similarity threshold quá cao (hiện tại: {similarity_threshold})")
                    print("   5. Keywords không khớp với tên entities trong database")
                    print("\n💡 Gợi ý:")
                    print(
                        "   - Kiểm tra xem dữ liệu có chứa 'xerostomia', 'cathepsin L', 'squamous cell carcinoma' không")
                    print("   - Thử giảm similarity threshold xuống 0.3-0.4")
                    print("   - Kiểm tra embeddings có được tạo từ cùng model không")

                # Hiển thị chi tiết TẤT CẢ Level 1 nodes
                if retrieval_result.get('level1_nodes'):
                    print("\n" + "=" * 80)
                    print("📊 CHI TIẾT TẤT CẢ LEVEL 1 NODES TỪ NEO4J")
                    print("=" * 80)
                    for i, node in enumerate(retrieval_result['level1_nodes'], 1):
                        print(f"\n--- Level 1 Node #{i} ---")
                        print(f"ID: {node.get('id', 'N/A')}")
                        print(f"Entity ID: {node.get('entity_id', 'N/A')}")
                        print(f"Name: {node.get('name', 'N/A')}")
                        print(f"Level: {node.get('level', 'N/A')}")
                        print(
                            f"Semantic Type: {node.get('semantic_type', 'N/A')}")
                        print(f"Entity Type: {node.get('entity_type', 'N/A')}")
                        print(
                            f"CUI: {node.get('cui', 'N/A') if node.get('cui') else 'N/A (empty)'}")
                        print(
                            f"ICD: {node.get('icd', 'N/A') if node.get('icd') else 'N/A (empty)'}")
                        definition = node.get('definition', '')
                        if definition:
                            print(f"Definition: {definition[:200]}..." if len(
                                definition) > 200 else f"Definition: {definition}")
                        else:
                            print(f"Definition: N/A (empty)")
                        description = node.get('description', '')
                        if description:
                            print(f"Description: {description[:200]}..." if len(
                                description) > 200 else f"Description: {description}")
                        else:
                            print(f"Description: N/A (empty)")
                        print(
                            f"Similarity Score: {node.get('similarity_score', 0):.4f}")
                        # Hiển thị tất cả keys để xem có thông tin gì khác không
                        all_keys = set(node.keys())
                        known_keys = {'id', 'entity_id', 'name', 'level', 'semantic_type',
                                      'cui', 'definition', 'icd', 'entity_type', 'description', 'similarity_score'}
                        other_keys = all_keys - known_keys
                        if other_keys:
                            print(f"Other fields: {other_keys}")
                            for key in other_keys:
                                print(f"  {key}: {node.get(key, 'N/A')}")
                else:
                    print("\n⚠️  Không có Level 1 nodes")

                # Hiển thị chi tiết TẤT CẢ Level 2 nodes
                if retrieval_result.get('level2_nodes'):
                    print("\n" + "=" * 80)
                    print("📊 CHI TIẾT TẤT CẢ LEVEL 2 NODES TỪ NEO4J")
                    print("=" * 80)
                    for i, node in enumerate(retrieval_result['level2_nodes'], 1):
                        print(f"\n--- Level 2 Node #{i} ---")
                        print(f"ID: {node.get('id', 'N/A')}")
                        print(f"Entity ID: {node.get('entity_id', 'N/A')}")
                        print(f"Name: {node.get('name', 'N/A')}")
                        print(f"CUI: {node.get('cui', 'N/A')}")
                        print(f"ICD: {node.get('icd', 'N/A')}")
                        print(f"Definition: {node.get('definition', 'N/A')[:200]}..." if len(node.get(
                            'definition', '')) > 200 else f"Definition: {node.get('definition', 'N/A')}")
                        semantic_types = node.get('semantic_types', [])
                        semantic_type = node.get('semantic_type', '')
                        print(f"Semantic Types (array): {semantic_types}")
                        print(f"Semantic Type (single): {semantic_type}")
                        print(f"Entity Type: {node.get('entity_type', 'N/A')}")
                        print(f"Description: {node.get('description', 'N/A')[:200]}..." if len(node.get(
                            'description', '')) > 200 else f"Description: {node.get('description', 'N/A')}")
                        # Hiển thị tất cả keys để xem có thông tin gì khác không
                        all_keys = set(node.keys())
                        known_keys = {'id', 'entity_id', 'name', 'cui', 'icd', 'definition',
                                      'semantic_types', 'semantic_type', 'entity_type', 'description'}
                        other_keys = all_keys - known_keys
                        if other_keys:
                            print(f"Other fields: {other_keys}")
                            for key in other_keys:
                                print(f"  {key}: {node.get(key, 'N/A')}")
                else:
                    print("\n⚠️  Không có Level 2 nodes")

                # Hiển thị chi tiết Relationships
                if retrieval_result.get('relationships'):
                    print("\n" + "=" * 80)
                    print("🔗 CHI TIẾT RELATIONSHIPS")
                    print("=" * 80)
                    # Hiển thị 10 đầu tiên
                    for i, rel in enumerate(retrieval_result.get('relationships', [])[:10], 1):
                        print(f"\n--- Relationship #{i} ---")
                        print(f"Source ID: {rel.get('source_id', 'N/A')}")
                        print(f"Source Name: {rel.get('source_name', 'N/A')}")
                        print(f"Target ID: {rel.get('target_id', 'N/A')}")
                        print(f"Target Name: {rel.get('target_name', 'N/A')}")
                        print(f"Type: {rel.get('type', 'N/A')}")
                        print(f"Description: {rel.get('description', 'N/A')}")
                        # Hiển thị tất cả keys
                        all_keys = set(rel.keys())
                        known_keys = {
                            'source_id', 'source_name', 'target_id', 'target_name', 'type', 'description'}
                        other_keys = all_keys - known_keys
                        if other_keys:
                            print(f"Other fields: {other_keys}")
                            for key in other_keys:
                                print(f"  {key}: {rel.get(key, 'N/A')}")
                    if len(retrieval_result.get('relationships', [])) > 10:
                        print(
                            f"\n... và {len(retrieval_result.get('relationships', [])) - 10} relationships khác")

                # Kiểm tra xem có nodes liên quan đến query không
                query_terms = ["xerostomia",
                               "cathepsin", "squamous", "carcinoma"]
                found_relevant = False
                relevant_nodes = []
                if retrieval_result.get('level1_nodes'):
                    for node in retrieval_result.get('level1_nodes', []):
                        node_name = node.get('name', '').lower()
                        if any(term in node_name for term in query_terms):
                            found_relevant = True
                            relevant_nodes.append(node.get('name', 'Unknown'))

                if not found_relevant and l1_nodes > 0:
                    print("\n" + "=" * 80)
                    print(
                        "⚠️  CẢNH BÁO: Tìm thấy nodes nhưng KHÔNG liên quan đến query")
                    print("=" * 80)
                    print("   Query terms tìm kiếm:", query_terms)
                    print("   Các nodes được tìm thấy (không khớp):")
                    for i, node in enumerate(retrieval_result.get('level1_nodes', []), 1):
                        print(
                            f"   {i}. {node.get('name', 'Unknown')} (ID: {node.get('id', 'N/A')})")
                    print("\n💡 Có thể do:")
                    print(
                        "   - Keywords được trích xuất không khớp với tên entities trong database")
                    print(
                        "   - Embeddings của keywords không tương đồng với embeddings của entities")
                    print(
                        "   - Cần cải thiện keyword extraction hoặc sử dụng exact matching")
                    print(
                        "   - Có thể cần tìm kiếm bằng tên chính xác thay vì embeddings")
                elif found_relevant:
                    print("\n✅ Tìm thấy nodes liên quan:")
                    for node_name in relevant_nodes:
                        print(f"   - {node_name}")

                # Format kết quả
                if l1_nodes > 0 or l2_nodes > 0:
                    formatted_text = format_retrieval_results(
                        level1_nodes=retrieval_result.get('level1_nodes', []),
                        level2_nodes=retrieval_result.get('level2_nodes', []),
                        relationships=retrieval_result.get('relationships', [])
                    )
                    result["formatted_text"] = formatted_text

                    print("\n📄 Đã format kết quả (độ dài: {} ký tự)".format(
                        len(formatted_text)))
                    preview = formatted_text[:500] if len(
                        formatted_text) > 500 else formatted_text
                    print(f"📝 Preview (500 ký tự đầu):\n{preview}...")

                    # Lưu kết quả vào file
                    saved_file = save_retrieval_result(
                        query, formatted_text, result, conversation_history)
                    if saved_file:
                        result["saved_file"] = saved_file
                        print(f"💾 Đã lưu prompt vào: {saved_file}")
                        print("   📋 File chứa prompt sẵn sàng để đưa vào LLM")

                    # BƯỚC 4: Tạo câu trả lời từ LLM
                    print("\n" + "-" * 80)
                    print("🤖 BƯỚC 4: TẠO CÂU TRẢ LỜI TỪ LLM GEMINI")
                    print("-" * 80)

                    # Format conversation history nếu có
                    history_text = None
                    if conversation_history:
                        history_text = "\n".join([
                            f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}"
                            for msg in conversation_history[-3:]
                        ])

                    llm_answer = await generate_answer_with_llm(
                        query=query,
                        formatted_text=formatted_text,
                        gemini_client=gemini_client,
                        conversation_history=history_text
                    )

                    if llm_answer:
                        result["llm_answer"] = llm_answer

                        # Hiển thị câu trả lời
                        print("\n✅ Câu trả lời từ LLM:")
                        print("=" * 80)
                        # Hiển thị preview nếu quá dài
                        if len(llm_answer) > 1000:
                            print(llm_answer[:1000])
                            print(
                                f"\n... (còn {len(llm_answer) - 1000} ký tự)")
                        else:
                            print(llm_answer)
                        print("=" * 80)

                        # Lưu câu trả lời vào file
                        answer_file = save_llm_answer(
                            query, llm_answer, result, saved_file)
                        if answer_file:
                            result["answer_file"] = answer_file
                            print(f"\n💾 Đã lưu câu trả lời vào: {answer_file}")
                    else:
                        print("⚠️  Không thể tạo câu trả lời từ LLM")

            except Exception as e:
                print(f"❌ Lỗi truy vấn đồ thị: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️  Không có keywords để truy vấn")
    else:
        print(
            f"\n⏭️  Intent là {intent.name} - Bỏ qua trích xuất keywords và truy vấn đồ thị")
        print("   (Chỉ xử lý cho HEALTHCARE_RELATED)")

    # Tổng kết
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Intent: {result['intent'].name if result['intent'] else 'N/A'}")
    if result['intent'] == QueryIntent.HEALTHCARE_RELATED:
        print(f"High-level keywords: {len(result['high_level_keywords'])}")
        print(f"Low-level keywords: {len(result['low_level_keywords'])}")
        if result['retrieval_result']:
            print(
                f"Level 1 nodes: {len(result['retrieval_result'].get('level1_nodes', []))}")
            print(
                f"Level 2 nodes: {len(result['retrieval_result'].get('level2_nodes', []))}")
            print(
                f"Relationships: {len(result['retrieval_result'].get('relationships', []))}")
    print("=" * 80)

    return result


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
