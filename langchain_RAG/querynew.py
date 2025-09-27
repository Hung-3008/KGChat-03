from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()
import logging, time
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

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


def load_llm_gemini_with_manager(
    key_manager: GeminiKeyManager,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.5,
    max_output_tokens: int = 512,
    max_retries: int = 6
) -> ChatGoogleGenerativeAI:
    attempt = 0
    while attempt < max_retries:
        api_key = key_manager.get_current_key() or key_manager.rotate_key()
        if not api_key:
            raise RuntimeError("Không tìm thấy API key hợp lệ cho Gemini.")

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            _ = llm.invoke("ping")  
            return llm

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

    raise RuntimeError(f"Không khởi tạo được Gemini LLM sau {max_retries} lần thử.")

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know. Do not make things up.
Only use information from the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(tpl: str):
    return PromptTemplate(template=tpl, input_variables=["context", "question"])
manager = GeminiKeyManager(current_key_index=1, max_keys=14)  # chỉnh max_keys theo số key bạn có




api_key = os.getenv("GOOGLE_API_KEY")
embed_model = os.getenv("EMBED_MODEL", "text-embedding-001")
embedding= GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)
client = QdrantClient(url="http://localhost:6333")
db = QdrantVectorStore.from_existing_collection(
    embedding=embedding,                 
    collection_name="database2",
    url="http://localhost:6333",         # kết nối HTTP
    content_payload_key="text",          # payload chứa nội dung chunk
    prefer_grpc=False                    # để chắc dùng HTTP
)
llm = load_llm_gemini_with_manager(manager, model_name="gemini-2.0-flash")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
)
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print(response["result"])
print("\n=== SOURCES ===")
for i, doc in enumerate(response["source_documents"], 1):
    meta = doc.metadata
    print(f"[{i}] doc={meta.get('doc')} | chunk={meta.get('chunk')}")
    print(doc.page_content[:200].replace("\n", " "), "\n")










# DB_FAISS_PATH = "./vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     encode_kwargs={"normalize_embeddings": True},
# )
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# llm = load_llm_gemini("gemini-2.0-flash")  
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": 10}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
# )