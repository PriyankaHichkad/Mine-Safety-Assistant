import os
import pickle
from typing import List, Dict, Any
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

class MineMindIndexer:
    """
    Batched Dual Indexer for Large PDF Libraries:
    1. Qdrant Local Embedded Vector DB (Batched vector upserts)
    2. BM25 Lexical Index
    """
    def __init__(
        self,
        db_path: str = "./data/qdrant_db",
        collection_name: str = "mining_knowledge",
        embedding_model: str = "BAAI/bge-base-en-v1.5"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        
        os.makedirs(db_path, exist_ok=True)
        try:
            self.qdrant = QdrantClient(path=db_path)
        except Exception:
            lock_file = os.path.join(db_path, ".lock")
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except Exception:
                    pass
            try:
                self.qdrant = QdrantClient(path=db_path)
            except Exception:
                self.qdrant = QdrantClient(location=":memory:")
        
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        vector_size = self.embedder.get_sentence_embedding_dimension()

        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {self.collection_name} (Dim: {vector_size})")

    def index_chunks(self, chunks: List[Dict[str, Any]], bm25_save_path: str = "./data/bm25_index.pkl", batch_size: int = 64):
        if not chunks:
            print("No chunks to index.")
            return

        print(f"Indexing {len(chunks)} chunks into Qdrant Local DB (Batch Size: {batch_size})...")
        
        contents = [c["content"] for c in chunks]
        
        # Batched Embeddings Generation
        print("Generating dense vector embeddings...")
        embeddings = self.embedder.encode(
            contents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        points = []
        tokenized_corpus = []

        print("Building Qdrant PointStruct objects & BM25 tokens...")
        for idx, chunk in enumerate(tqdm(chunks, desc="Processing Points")):
            vector = embeddings[idx].tolist()
            content = chunk["content"]
            metadata = chunk["metadata"]
            
            points.append(PointStruct(
                id=idx + 1,
                vector=vector,
                payload={
                    "chunk_id": chunk["id"],
                    "content": content,
                    **metadata
                }
            ))
            
            tokenized_corpus.append(content.lower().split())

        # Upsert in batches into Qdrant
        print(f"Upserting points into Qdrant Local DB at {self.db_path}...")
        for i in range(0, len(points), 500):
            batch_points = points[i:i+500]
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )

        print(f"Successfully upserted {len(points)} points into Qdrant.")

        # Build & Save BM25 Index
        print("Building BM25 lexical index...")
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_data = {
            "bm25": bm25,
            "chunks": chunks
        }
        with open(bm25_save_path, "wb") as f:
            pickle.dump(bm25_data, f)
        print(f"Successfully saved BM25 index to {bm25_save_path}.")
        
        # Release Qdrant DB lock
        try:
            self.qdrant.close()
        except Exception:
            pass
