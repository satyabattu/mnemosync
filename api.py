"""
FastAPI backend for MnemoSync.
Auto-generated docs at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from classifier.intent_classifier import IntentClassifier
from drift.drift_engine import compute_drift_timeline
from rag.memory_store import MemoryStore
from rag.retrieval_engine import load_persona_into_store, query as retrieve
from rag.conflict_resolver import detect_contradictions, resolve_conflicts

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] Loading models and data...")
    _state["clf"] = IntentClassifier()
    _state["store"] = MemoryStore()
    load_persona_into_store(_state["store"])
    _state["store"].rebuild_index()
    print("[api] Ready")
    yield


app = FastAPI(
    title="MnemoSync API",
    description="Local-first memory integrity engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 6


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify_intent(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    return _state["clf"].predict(req.text)


@app.get("/drift/timeline")
def get_drift_timeline():
    timeline = compute_drift_timeline()
    return {"timeline": timeline, "total_days": len(timeline)}


@app.post("/memory/query")
def query_memory(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    chunks = retrieve(_state["store"], req.query, top_k=req.top_k)
    contradictions = detect_contradictions(chunks)
    resolution = resolve_conflicts(req.query, chunks, contradictions)

    return {
        "query": req.query,
        "retrieved_chunks": len(chunks),
        "resolution": resolution,
    }


@app.get("/memory/all")
def get_all_memory():
    return {"messages": _state["store"].get_all()}