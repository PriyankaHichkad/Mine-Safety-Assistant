import os
import sys
import time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.hybrid_search import MineMindHybridRetriever
from src.generation.rag_engine import MineMindRAGEngine

app = FastAPI(
    title="MineMind Mining AI & Regulatory Compliance API",
    description="Enterprise Hybrid RAG + Observability Telemetry API for Mining Engineering",
    version="1.0.0"
)

# Enable CORS for Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global telemetry logs
QUERY_TELEMETRY_LOGS: List[Dict[str, Any]] = []

# Global RAG Engine instance
retriever = None
rag_engine = None

@app.on_event("startup")
def startup_event():
    global retriever, rag_engine
    print("Initializing MineMind Hybrid Retriever & Qdrant Engine...")
    db_path = "./data/qdrant_db"
    bm25_path = "./data/bm25_index.pkl"
    
    if os.path.exists(db_path) and os.path.exists(bm25_path):
        retriever = MineMindHybridRetriever(db_path=db_path, bm25_path=bm25_path)
        rag_engine = MineMindRAGEngine(retriever)
        print("MineMind RAG Engine ready!")
    else:
        print("WARNING: Qdrant DB or BM25 index not found. Please run scripts/ingest_docs.py first.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 15
    top_m: int = 3

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "system": "MineMind Enterprise RAG Engine",
        "qdrant_db": os.path.exists("./data/qdrant_db"),
        "bm25_index": os.path.exists("./data/bm25_index.pkl")
    }

@app.post("/api/search")
def search_mining_docs(req: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized. Run ingestion script.")
    
    start_time = time.time()
    results = retriever.search(query=req.query, top_k=req.top_k, final_top_m=req.top_m)
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "query": req.query,
        "results_count": len(results),
        "latency_ms": round(latency_ms, 2),
        "results": results
    }

@app.post("/api/query")
def query_rag_engine(req: QueryRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized. Run ingestion script.")
    
    response = rag_engine.answer_query(req.query)
    
    # Store telemetry log
    QUERY_TELEMETRY_LOGS.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": req.query,
        "telemetry": response["telemetry"]
    })
    
    return response

@app.get("/api/telemetry")
def get_telemetry_metrics():
    if not QUERY_TELEMETRY_LOGS:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "queries": []
        }
    
    latencies = [log["telemetry"]["total_latency_ms"] for log in QUERY_TELEMETRY_LOGS]
    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    
    return {
        "total_queries": len(QUERY_TELEMETRY_LOGS),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted_latencies[min(p95_index, len(sorted_latencies)-1)], 2),
        "recent_queries": QUERY_TELEMETRY_LOGS[-10:]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
