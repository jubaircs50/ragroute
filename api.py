"""Minimal FastAPI retrieval endpoint."""
from __future__ import annotations
import os
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ragroute.data import load_corpus_jsonl
from ragroute.demo import DEMO_DOCUMENTS
from ragroute.retrieval import RetrievalEngine
load_dotenv()
app = FastAPI(title="RAGRoute API", description="Adaptive retrieval API", version="0.1.0")
_engine: RetrievalEngine | None = None
_lock = Lock()
class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    mode: str = Field(default="adaptive", pattern="^(bm25|dense|hybrid|adaptive)$")
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = False

def get_engine() -> RetrievalEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                path_text = os.getenv("RAGROUTE_CORPUS_PATH", "").strip()
                if path_text:
                    path = Path(path_text).expanduser()
                    if not path.is_file(): raise RuntimeError(f"Corpus file not found: {path}")
                    docs = load_corpus_jsonl(path.read_text(encoding="utf-8-sig"))
                else: docs = DEMO_DOCUMENTS
                _engine = RetrievalEngine(docs)
    return _engine
@app.get("/health")
def health() -> dict[str, str]: return {"status":"ok"}
@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict:
    try: return get_engine().search(request.query, request.mode, request.top_k, request.rerank)
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc
