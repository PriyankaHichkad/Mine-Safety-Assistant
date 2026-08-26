import os
import pickle
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder

class MineMindHybridRetriever:
    """
    Hybrid Retriever combining:
    1. Lexical BM25 keyword matching
    2. Qdrant Vector semantic search (File DB or In-Memory Cloud Client)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Reranking
    """
    def __init__(
        self,
        db_path: str = "./data/qdrant_db",
        collection_name: str = "mining_knowledge",
        bm25_path: str = "./data/bm25_index.pkl",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(embedding_model)
        
        # Load BM25 Index
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.bm25_chunks = bm25_data["chunks"]

        # Initialize Qdrant Client (File DB with In-Memory Cloud Fallback)
        use_memory = False
        try:
            self.qdrant = QdrantClient(path=db_path)
            # Test collection access
            cols = [c.name for c in self.qdrant.get_collections().collections]
            if collection_name not in cols:
                use_memory = True
        except Exception as err:
            print(f"[Qdrant Client Notice]: File DB locked or unavailable ({err}). Switching to in-memory Qdrant client...")
            use_memory = True

        if use_memory:
            self.qdrant = QdrantClient(location=":memory:")
            self._build_in_memory_vector_db()

        # Load Reranker Model
        print(f"Loading cross-encoder reranker: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model)

    def _build_in_memory_vector_db(self):
        """Populates in-memory Qdrant database from BM25 chunks for cloud deployment."""
        print("Populating in-memory Qdrant vector database...")
        vector_size = self.embedder.get_sentence_embedding_dimension()
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        contents = [c["content"] for c in self.bm25_chunks]
        embeddings = self.embedder.encode(contents, show_progress_bar=False, convert_to_numpy=True)
        
        points = []
        for idx, chunk in enumerate(self.bm25_chunks):
            points.append(PointStruct(
                id=idx + 1,
                vector=embeddings[idx].tolist(),
                payload={
                    "chunk_id": chunk["id"],
                    "content": chunk["content"],
                    **chunk["metadata"]
                }
            ))
            
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"In-memory Qdrant vector database ready with {len(points)} points!")

    def search(self, query: str, top_k: int = 15, final_top_m: int = 5) -> List[Dict[str, Any]]:
        # 1. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        bm25_results = []
        for rank, idx in enumerate(bm25_top_indices):
            chunk = self.bm25_chunks[idx]
            bm25_results.append({
                "chunk": chunk,
                "rank": rank + 1,
                "score": float(bm25_scores[idx])
            })

        # 2. Qdrant Vector Search
        query_vector = self.embedder.encode(query).tolist()
        try:
            if hasattr(self.qdrant, "query_points"):
                res = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k
                )
                qdrant_points = res.points if hasattr(res, "points") else res
            else:
                qdrant_points = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
        except Exception:
            qdrant_points = []
        
        vector_results = []
        for rank, point in enumerate(qdrant_points):
            vector_results.append({
                "chunk": {
                    "id": point.payload["chunk_id"],
                    "content": point.payload["content"],
                    "metadata": {k: v for k, v in point.payload.items() if k not in ["chunk_id", "content"]}
                },
                "rank": rank + 1,
                "score": float(point.score)
            })

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        chunk_map = {}
        rrf_k = 60

        for res in bm25_results:
            c_id = res["chunk"]["id"]
            chunk_map[c_id] = res["chunk"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + res["rank"]))

        for res in vector_results:
            c_id = res["chunk"]["id"]
            chunk_map[c_id] = res["chunk"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + res["rank"]))

        fused_candidates = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        # 4. Cross-Encoder Reranking
        rerank_pairs = [[query, chunk_map[cid]["content"]] for cid in fused_candidates]
        cross_scores = self.reranker.predict(rerank_pairs)

        reranked_results = []
        for idx, cid in enumerate(fused_candidates):
            reranked_results.append({
                "chunk": chunk_map[cid],
                "rrf_score": float(rrf_scores[cid]),
                "rerank_score": float(cross_scores[idx])
            })

        reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)[:final_top_m]
        return reranked_results

    def close(self):
        try:
            if hasattr(self, "qdrant") and self.qdrant:
                self.qdrant.close()
        except Exception:
            pass
