import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.query.baseline import RetrievalManager

 
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

import json
from fastapi.responses import StreamingResponse
import asyncio
 
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received question: {request.question}")
    
    async def event_generator():
        try:
            for step_data in retrieval_manager.run_for_frontend_stream(request.question, request.grounding):
                yield f"data: {json.dumps(step_data)}\n\n"
                await asyncio.sleep(0.1) 
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
