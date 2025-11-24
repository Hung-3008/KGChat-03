import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.query.baseline import RetrievalManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("API")

app = FastAPI(title="Medical Chatbot API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RetrievalManager
try:
    retrieval_manager = RetrievalManager()
except Exception as e:
    logger.error(f"Failed to initialize RetrievalManager: {e}")
    raise

class ChatRequest(BaseModel):
    question: str
    grounding: bool = True

class ChatResponse(BaseModel):
    answer: str
    steps: Dict[str, Any]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received question: {request.question}")
    try:
        result = retrieval_manager.run_for_frontend(request.question, request.grounding)
        return ChatResponse(
            answer=result["final_answer"],
            steps=result
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
