import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.chunker import MiningDocumentChunker
from src.ingestion.indexer import MineMindIndexer

def main():
    search_directories = ["./data/sample_docs", "./data/pdf_books", "./data/msha_reports", "./data/weebly_books"]
    db_path = "./data/qdrant_db"
    bm25_path = "./data/bm25_index.pkl"

    print("=== MineMind Large Corpus Ingestion Pipeline ===")
    chunker = MiningDocumentChunker()
    
    all_chunks = []
    
    for d in search_directories:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            continue
            
        print(f"\nScanning directory: {d}...")
        files = [f for f in os.listdir(d) if f.endswith(".txt") or f.endswith(".pdf")]
        print(f"Found {len(files)} documents in {d}.")
        
        for filename in files:
            filepath = os.path.join(d, filename)
            print(f"  -> Processing: {filename}...")
            try:
                chunks = chunker.process_file(filepath)
                all_chunks.extend(chunks)
                print(f"     [Extracted {len(chunks)} semantic chunks]")
            except Exception as err:
                print(f"     [Error processing {filename}: {err}]")

    print(f"\nTotal extracted chunks across all library documents: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks extracted from PDF books. Fallback: Generating official MSHA/OSHA reports...")
        from scripts.scrape_msha_accidents import generate_msha_dataset
        generate_msha_dataset()
        for d in ["./data/msha_reports"]:
            files = [f for f in os.listdir(d) if f.endswith(".txt")]
            for filename in files:
                filepath = os.path.join(d, filename)
                chunks = chunker.process_file(filepath)
                all_chunks.extend(chunks)

    # Index into Qdrant & BM25
    indexer = MineMindIndexer(db_path=db_path, collection_name="mining_knowledge")
    indexer.index_chunks(all_chunks, bm25_save_path=bm25_path, batch_size=64)
    print("\n=== Large Corpus Ingestion Complete! ===")

if __name__ == "__main__":
    main()
