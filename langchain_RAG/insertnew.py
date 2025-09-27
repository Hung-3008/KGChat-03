import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import time
from tqdm import tqdm 
import re
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import uuid
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiKeyManager:
    def __init__(
        self,
        current_key_index: int = 1,
        max_keys: int = 6,
        key_pattern: str = "GEMINI_API_KEY_{}"
    ):
        self.current_key_index = current_key_index
        self.max_keys = max_keys
        self.key_pattern = key_pattern
        self.logger = logging.getLogger(__name__)
    
    def get_current_key(self) -> Optional[str]:
        key_name = self.key_pattern.format(self.current_key_index)
        api_key = os.getenv(key_name)
        
        if not api_key:
            self.logger.warning(f"API key {key_name} not found in environment variables")
            return None
            
        return api_key
    
    def rotate_key(self) -> Optional[str]:

        for _ in range(self.max_keys):

            self.current_key_index = (self.current_key_index % self.max_keys) + 1
            
            key_name = self.key_pattern.format(self.current_key_index)
            api_key = os.getenv(key_name)
            
            if api_key:
                self.logger.info(f"Rotated to API key {key_name}")
                return api_key
        
        self.logger.error("No valid API keys found after trying all options")
        return None

def batch_embed_with_manager(
    key_manager: GeminiKeyManager,
    chunks: list[str],
    model_name: str,
    batch_size: int = 12,
    sleep: float = 2.0,
    max_retries_per_batch: int = 6,
):
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        success = False
        attempt = 0

        while not success and attempt < max_retries_per_batch:
            api_key = key_manager.get_current_key() or key_manager.rotate_key()
            if not api_key:
                raise RuntimeError("Không tìm thấy API key hợp lệ cho Gemini.")

            embedder = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            try:
                vecs = embedder.embed_documents(batch)
                embeddings.extend(vecs)
                success = True
            except Exception as e:
                msg = str(e).lower()
                attempt += 1
                if "429" in msg or "quota" in msg:
                    logger.warning(f"Key {api_key[:6]}... quota exceeded → rotate.")
                    key_manager.rotate_key()
                    time.sleep(min(2.0 * attempt, 10.0))
                    continue
                if "403" in msg or "suspended" in msg:
                    logger.error(f"Key {api_key[:6]}... suspended → skip.")
                    key_manager.rotate_key()
                    continue
                raise

        if not success:
            raise RuntimeError(f"Batch {i//batch_size} thất bại sau {max_retries_per_batch} lần thử.")

        time.sleep(sleep)  # giảm burst

    return embeddings

DATA_DIR = r"C:\Users\NC\Downloads\NEO4Jsetup\123\datatext"
COLLECTION = "database2"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-001")  # nên đồng bộ '004'

client = QdrantClient(url="http://localhost:6333")
try:
    client.get_collection(COLLECTION)
    client.delete_collection(COLLECTION)
except Exception:
    pass


client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
manager = GeminiKeyManager(current_key_index=1, max_keys=14)


folder = Path(DATA_DIR)
txt_files = sorted(folder.glob("*.txt"))

BATCH_UPSERT = 512
points_buf = []
total_chunks = 0

for fp in tqdm(txt_files, desc="Processing files"):
    try:
        text = fp.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Read error {fp.name}: {e}")
        continue

    chunks = splitter.split_text(text)
    if not chunks:
        continue


    vecs = batch_embed_with_manager(
        key_manager=manager,
        chunks=chunks,
        model_name=EMBED_MODEL,
        batch_size=12,
        sleep=2.0
    )


    if len(vecs) != len(chunks):
        raise RuntimeError(f"Embedding count mismatch for {fp.name}: {len(vecs)} vs {len(chunks)}")

    for idx, (ch, v) in enumerate(zip(chunks, vecs), start=1):
        pid = str(uuid.uuid4())
        points_buf.append(PointStruct(
            id=pid,
            vector=v,
            payload={
                "text": ch,          
                "doc": fp.stem,
                "chunk": idx
            }
        ))
        total_chunks += 1

        if len(points_buf) >= BATCH_UPSERT:
            client.upsert(collection_name=COLLECTION, points=points_buf)
            points_buf.clear()


if points_buf:
    client.upsert(collection_name=COLLECTION, points=points_buf)
print(f"✅ Đã insert {total_chunks} vectors vào Qdrant collection '{COLLECTION}'")

# DATA_DIR = r"C:\Users\NC\Downloads\NEO4Jsetup\123\datatext"
# folder = Path(DATA_DIR)
# txt_files = sorted(folder.glob("*.txt"))

# chunk_records = []
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# for fp in txt_files:
#     with open(fp, "r", encoding="utf-8") as f:
#         text = f.read()
#     chunks = splitter.split_text(text)
#     for idx, ch in enumerate(chunks, start=1):
#         doc_name = fp.stem
#         chunk_records.append((doc_name, 1, idx, ch))



# manager = GeminiKeyManager(current_key_index=1, max_keys=14)  
# model_name = os.getenv("EMBED_MODEL")

# all_texts = [t for (_, _, _, t) in chunk_records]
# embeddings = batch_embed_with_manager(
#     key_manager=manager,
#     chunks=all_texts,
#     model_name=model_name,
#     batch_size=8,  
#     sleep=0.5
# )
# dim = len(embeddings[0])
# # Qdrant
# client = QdrantClient(url="http://localhost:6333")
# COLLECTION = "database1"

# client.recreate_collection(
#     collection_name=COLLECTION,
#     vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
# )

# # Upsert
# points = [
#     PointStruct(
#         id=str(uuid.uuid4()),
#         vector=vec,
#         payload={"text": ch, "doc": doc, "page": page, "chunk": idx}  # <-- lưu text và metadata
#     )
#     for (doc, page, idx, ch), vec in zip(chunk_records, embeddings)
# ]
# client.upsert(collection_name=COLLECTION, points=points)
# print(f"✅ Đã insert {len(points)} vectors vào Qdrant collection '{COLLECTION}'")
















# def get_embed():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},                    # hoặc "cuda" nếu có GPU
#         encode_kwargs={"normalize_embeddings": True}       # rất nên bật
#     )
#     return embedding_model







# api_keys = [k.strip() for k in os.getenv("GOOGLE_API_KEY", "").split(",")]
# if not api_keys:
#     raise ValueError("Không có GOOGLE_API_KEY trong .env")

# # Tạo vòng xoay key
# key_cycle = itertools.cycle(api_keys)

# def get_embedder_with_key(key: str, model_name: str):
#     return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)

# def batch_embed(chunks, model_name="text-embedding-004", batch_size=32, sleep=0.5, max_retries=3):
#     embeddings = []
#     for i in tqdm(range(0, len(chunks), batch_size)):
#         batch = chunks[i:i+batch_size]
#         success = False
#         retries = 0

#         while not success and retries < max_retries:
#             key = next(key_cycle)  # lấy key tiếp theo
#             embedder = get_embedder_with_key(key, model_name)
#             try:
#                 vecs = embedder.embed_documents(batch)
#                 embeddings.extend(vecs)
#                 success = True
#             except Exception as e:
#                 print(f"[Batch {i//batch_size}] Key {key[:6]}... bị lỗi: {e}")
#                 retries += 1
#                 time.sleep(2 * retries)  # backoff

#         if not success:
#             raise RuntimeError(f"Batch {i//batch_size} thất bại sau {max_retries} retries.")

#         time.sleep(sleep)  # delay giữa các batch

#     return embeddings




# print(f"📄 Tổng số chunks: {len(chunk_records)}")






# def get_embed():
#     model_name = os.getenv("EMBED_MODEL")
#     api_keys = os.getenv("GOOGLE_API_KEY")
#     if not api_keys:
#         raise ValueError("Thiếu GOOGLE_API_KEY cho embeddings (Gemini).")
#     if not model_name:
#         raise ValueError("Thiếu EMBED_MODEL cho embeddings (Gemini).")
#     api_keys = [k.strip() for k in api_keys.split(",")]
#     embedder = None
#     last_error = None
#     for key in api_keys:
#         try:
#             embedder = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)
#             print(f"Đang dùng GOOGLE_API_KEY: {key[:6]}...{key[-4:]}")
#             break
#         except Exception as e:
#             print(f"API key {key[:6]}... lỗi: {e}")
#             last_error = e
#             embedder = None
#     if embedder is None:
#         raise RuntimeError(f"Tất cả Google API key đều lỗi. Lỗi cuối: {last_error}")
#     return embedder 


# def batch_embed(embedder,chunks, batch_size=32, sleep=0.3):
#     embeddings = []
#     for i in tqdm(range(0, len(chunks), batch_size)):
#         batch = chunks[i:i+batch_size]
#         try:
#             vecs = embedder.embed_documents(batch)
#             embeddings.extend(vecs)
#         except Exception as e:
#             print(f"Lỗi ở batch {i//batch_size}: {e}")
#         time.sleep(sleep)  # để tránh rate limit
#     return embeddings


# # Nhúng theo batch
# embedder = get_embed()
# all_texts = [t for (_, _, _, t) in chunk_records]
# embeddings = batch_embed(embedder, all_texts, batch_size=32, sleep=0.3)
# embeddings = batch_embed(all_texts, model_name=os.getenv("EMBED_MODEL"), batch_size=16, sleep=0.5)