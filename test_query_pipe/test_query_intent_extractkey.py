"""
Test script để kiểm tra Query Intent Classification và Keyword Extraction
Test các chức năng:
1. analyze_query - phân loại intent từ query_analyzer.py
2. extract_keywords - trích xuất high-level và low-level keywords từ keyword_extractor.py
   (CHỈ trích xuất keywords khi intent là HEALTHCARE_RELATED)
"""
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

# Import các modules cần thiết
try:
    from backend.llm.providers.gemini.gemini_client import GeminiClient
    from backend.llm.providers.gemini.gemini_config import GeminiConfig
    from backend.pipeline.query_analyzer import analyze_query, QueryIntent
    from backend.pipeline.keyword_extractor import extract_keywords
    print("✅ Đã import các modules thành công")
except ImportError as e:
    print(f"❌ Lỗi import modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def initialize_gemini_client():
    """Khởi tạo Gemini Client"""
    print("=" * 60)
    print("🔧 KHỞI TẠO GEMINI CLIENT")
    print("=" * 60)

    try:
        gemini_config = GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY_16"),
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        client = GeminiClient(gemini_config)
        print("✅ Gemini Client đã được khởi tạo")
        return client
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Gemini Client: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_query_intent(query: str, conversation_history=None, client=None):
    """Test phân loại intent của query"""
    print("\n" + "=" * 60)
    print("📋 TEST PHÂN LOẠI INTENT")
    print("=" * 60)
    print(f"Query: {query}")

    try:
        intent = await analyze_query(
            query=query,
            conversation_history=conversation_history,
            client=client
        )

        print(f"\n✅ Intent được phân loại: {intent.name} ({intent.value})")
        return intent
    except Exception as e:
        print(f"❌ Lỗi khi phân loại intent: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_keyword_extraction(query: str, conversation_history=None, client=None):
    """Test trích xuất keywords từ query"""
    print("\n" + "=" * 60)
    print("🔑 TEST TRÍCH XUẤT KEYWORDS")
    print("=" * 60)
    print(f"Query: {query}")

    try:
        high_level_keywords, low_level_keywords = await extract_keywords(
            query=query,
            conversation_history=conversation_history,
            llm_client=client
        )

        print(f"\n✅ High-level keywords ({len(high_level_keywords)}):")
        for i, keyword in enumerate(high_level_keywords, 1):
            print(f"   {i}. {keyword}")

        print(f"\n✅ Low-level keywords ({len(low_level_keywords)}):")
        for i, keyword in enumerate(low_level_keywords, 1):
            print(f"   {i}. {keyword}")

        return high_level_keywords, low_level_keywords
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất keywords: {e}")
        import traceback
        traceback.print_exc()
        return [], []


async def test_full_pipeline(query: str, conversation_history=None, client=None):
    """Test toàn bộ pipeline: intent classification + keyword extraction (chỉ khi HEALTHCARE_RELATED)"""
    print("\n" + "=" * 80)
    print("🚀 TEST TOÀN BỘ PIPELINE")
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    # 1. Phân loại intent
    intent = await test_query_intent(query, conversation_history, client)

    # 2. Chỉ trích xuất keywords nếu intent là HEALTHCARE_RELATED
    high_level_keywords = []
    low_level_keywords = []

    if intent == QueryIntent.HEALTHCARE_RELATED:
        print("\n✅ Intent là HEALTHCARE_RELATED - Tiến hành trích xuất keywords...")
        high_level_keywords, low_level_keywords = await test_keyword_extraction(
            query, conversation_history, client
        )
    else:
        print(
            f"\n⏭️  Intent là {intent.name if intent else 'N/A'} - Bỏ qua trích xuất keywords (chỉ trích xuất cho HEALTHCARE_RELATED)")

    # 3. Tổng kết
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT")
    print("=" * 80)
    print(f"Intent: {intent.name if intent else 'N/A'}")
    if intent == QueryIntent.HEALTHCARE_RELATED:
        print(f"High-level keywords: {len(high_level_keywords)}")
        print(f"Low-level keywords: {len(low_level_keywords)}")
    else:
        print("Keywords: Không trích xuất (chỉ trích xuất cho HEALTHCARE_RELATED)")
    print("=" * 80)

    return {
        "intent": intent,
        "high_level_keywords": high_level_keywords,
        "low_level_keywords": low_level_keywords
    }


async def main():
    """Hàm main để chạy các test"""
    print("\n" + "=" * 80)
    print("🧪 TEST QUERY INTENT CLASSIFICATION & KEYWORD EXTRACTION")
    print("=" * 80)

    # Khởi tạo client
    client = initialize_gemini_client()
    if not client:
        print("❌ Không thể khởi tạo client. Dừng test.")
        return

    # Danh sách các query test
    test_queries = [
        # Healthcare queries
        "What are the symptoms of Type 2 diabetes?",
        "How does insulin work in the body?",
        "Can you tell me about diabetes complications?",

        # Greeting queries
        "Hello, how are you?",
        "Hi there!",

        # General queries
        "What is the weather today?",
        "Tell me a joke",

        # Personal info queries
        "My name is John and I'm 45 years old",
        "I have Type 2 diabetes and take metformin",
    ]

    # Chạy test cho từng query
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*80}")

        result = await test_full_pipeline(query, client=client)
        results.append({
            "query": query,
            "result": result
        })

        # Nghỉ một chút giữa các test để tránh rate limit
        if i < len(test_queries):
            await asyncio.sleep(2)

    # Tổng kết cuối cùng
    print("\n\n" + "=" * 80)
    print("📈 TỔNG KẾT TẤT CẢ CÁC TEST")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Query: {result['query']}")
        if result['result']['intent']:
            intent_name = result['result']['intent'].name
            print(f"   Intent: {intent_name}")
            if result['result']['intent'] == QueryIntent.HEALTHCARE_RELATED:
                print(
                    f"   High-level keywords: {len(result['result']['high_level_keywords'])}")
                print(
                    f"   Low-level keywords: {len(result['result']['low_level_keywords'])}")
            else:
                print(
                    "   Keywords: Không trích xuất (chỉ trích xuất cho HEALTHCARE_RELATED)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
