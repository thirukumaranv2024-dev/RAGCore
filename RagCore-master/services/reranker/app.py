"""Local cross-encoder reranking API for RAGCore.

Install: pip install -r requirements.txt
Run:     uvicorn app:app --host 127.0.0.1 --port 8001
The model downloads on first start; after caching, inference runs locally.
"""
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
app = FastAPI(title="RAGCore Local Cross-Encoder Reranker", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class Candidate(BaseModel):
    id: str
    text: str
    original_score: Optional[float] = None
    original_rank: Optional[int] = None

class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    candidates: List[Candidate] = Field(min_length=1, max_length=50)
    top_k: int = Field(default=5, ge=1, le=50)

@lru_cache(maxsize=1)
def get_model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME, max_length=512)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "execution": "local"}

@app.post("/rerank")
def rerank(request: RerankRequest):
    try:
        model = get_model()
        pairs = [(request.query, candidate.text) for candidate in request.candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(request.candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        return {
            "model": MODEL_NAME,
            "results": [
                {"id": candidate.id, "score": float(score)}
                for candidate, score in ranked[:request.top_k]
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {exc}") from exc
