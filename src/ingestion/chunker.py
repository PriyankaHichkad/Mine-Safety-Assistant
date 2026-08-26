import os
import re
import json
from typing import List, Dict, Any
from pypdf import PdfReader
from src.ingestion.multimodal_extractor import MiningMultimodalExtractor

class MiningDocumentChunker:
    """
    Multimodal Semantic Chunker for Mining Regulations (.txt) and PDF Textbooks (.pdf).
    Extracts text, page numbers, chapters, real author names, and embedded diagrams/figures.
    """
    def __init__(self, chunk_size: int = 600, overlap: int = 80, catalog_path: str = "./data/minemountain_catalog.json", author_map_path: str = "./data/book_authors.json"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.catalog_map = {}
        self.author_map = {}
        self.image_extractor = MiningMultimodalExtractor()
        
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                    for item in catalog:
                        self.catalog_map[item["filename"]] = item
            except Exception as e:
                pass

        if os.path.exists(author_map_path):
            try:
                with open(author_map_path, "r", encoding="utf-8") as f:
                    self.author_map = json.load(f)
            except Exception as e:
                pass

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        
        if filename.endswith(".pdf"):
            return self._process_pdf(file_path, filename)
        else:
            return self._process_txt(file_path, filename)

    def _process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        reader = PdfReader(file_path)
        catalog_info = self.catalog_map.get(filename, {})
        
        book_title = catalog_info.get("book_title") or filename.replace(".pdf", "").replace("_", " ")
        author = self.author_map.get(filename) or catalog_info.get("author") or "Mining Engineering Specialist"
        
        chunks = []
        global_chunk_id = 0

        # 1. Text Page-by-Page Extraction
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if not page_text or len(page_text.strip()) < 50:
                continue

            cleaned_text = self._clean_text(page_text)
            
            start = 0
            page_chunk_num = 0
            while start < len(cleaned_text):
                end = start + self.chunk_size
                chunk_str = cleaned_text[start:end]
                
                global_chunk_id += 1
                page_chunk_num += 1
                
                header_context = f"[Book: {book_title} | Author: {author} | Page {page_num}]"
                full_content = f"{header_context}\n{chunk_str}"
                
                chunks.append({
                    "id": f"{filename}_p{page_num}_c{page_chunk_num}",
                    "content": full_content,
                    "metadata": {
                        "source_file": filename,
                        "doc_title": book_title,
                        "author": author,
                        "page_number": page_num,
                        "category": "Mining Textbook / E-Library",
                        "section": f"Page {page_num}"
                    }
                })
                
                start += self.chunk_size - self.overlap

        # 2. Extract Embedded Diagrams & Figures
        diagram_chunks = self.image_extractor.extract_page_diagrams(reader, filename)
        for dc in diagram_chunks:
            dc["metadata"]["author"] = author
            dc["metadata"]["doc_title"] = book_title
        chunks.extend(diagram_chunks)

        return chunks

    def _process_txt(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_title = self._extract_field(text, "DOCUMENT_TITLE", filename)
        category = self._extract_field(text, "CATEGORY", "Indian Mining Legislation & Safety")
        publisher = self._extract_field(text, "PUBLISHER", self._extract_field(text, "ISSUER", "DGMS / Govt. of India"))
        author = self.author_map.get(doc_title, publisher)

        section_pattern = r"(--- (?:REGULATION|SECTION) \d+:[^\n]+---)"
        parts = re.split(section_pattern, text)

        chunks = []
        current_section = "General Overview"
        chunk_id = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part.startswith("--- REGULATION") or part.startswith("--- SECTION"):
                current_section = part.strip("- ").strip()
                continue

            paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                if "DOCUMENT_TITLE:" in para or "CATEGORY:" in para or "PUBLISHER:" in para:
                    continue

                chunk_id += 1
                chunks.append({
                    "id": f"{filename}_{chunk_id}",
                    "content": f"[{doc_title} | Author: {author} | {current_section}]\n{para}",
                    "metadata": {
                        "source_file": filename,
                        "doc_title": doc_title,
                        "author": author,
                        "category": category,
                        "section": current_section,
                        "paragraph_id": p_idx + 1
                    }
                })

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_field(self, text: str, field_name: str, default_val: str) -> str:
        match = re.search(rf"{field_name}:\s*([^\n]+)", text)
        return match.group(1).strip() if match else default_val
