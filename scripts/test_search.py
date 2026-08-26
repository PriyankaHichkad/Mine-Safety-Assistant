import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_search import MineMindHybridRetriever

def main():
    print("=== Testing MineMind Hybrid Search Engine ===")
    retriever = MineMindHybridRetriever(
        db_path="./data/qdrant_db",
        collection_name="mining_knowledge",
        bm25_path="./data/bm25_index.pkl"
    )

    test_queries = [
        "What is the maximum permissible methane gas percentage under CMR 1957 / 2017 Regulation 104?",
        "What are the haul road slope and gradient standards in DGMS circulars?",
        "How is Rock Mass Rating RMR used for roof bolting support standards?"
    ]

    for q in test_queries:
        print(f"\nQUERY: {q}")
        print("-" * 60)
        results = retriever.search(q, top_k=10, final_top_m=3)
        for i, res in enumerate(results, 1):
            chunk = res["chunk"]
            print(f"[{i}] Rerank Score: {res['rerank_score']:.4f} | Source: {chunk['metadata']['source_file']}")
            print(f"    Section: {chunk['metadata']['section']}")
            print(f"    Snippet: {chunk['content'][:200]}...")
            print("-" * 40)

if __name__ == "__main__":
    main()
