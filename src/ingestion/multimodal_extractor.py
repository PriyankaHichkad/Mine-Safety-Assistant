import os
import io
from typing import List, Dict, Any
from pypdf import PdfReader

class MiningMultimodalExtractor:
    """
    Multimodal Image & Diagram Extractor for Mining Textbooks.
    Extracts embedded diagrams/figures from PDF pages, saves image artifacts,
    and generates visual context descriptions for indexing.
    """
    def __init__(self, output_img_dir: str = "./data/extracted_images"):
        self.output_img_dir = output_img_dir
        os.makedirs(self.output_img_dir, exist_ok=True)

    def extract_page_diagrams(self, reader: PdfReader, filename: str) -> List[Dict[str, Any]]:
        extracted_diagrams = []
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                for img_idx, image_obj in enumerate(page.images, 1):
                    img_name = f"{filename}_p{page_num}_fig{img_idx}.png"
                    img_path = os.path.join(self.output_img_dir, img_name)
                    
                    # Save image bytes if not existing
                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f:
                            f.write(image_obj.data)
                        
                    caption = (
                        f"[Diagram/Figure in {filename}, Page {page_num}]: "
                        f"Technical mining schematic / chart extracted from page {page_num}."
                    )
                    
                    extracted_diagrams.append({
                        "id": f"{filename}_p{page_num}_img{img_idx}",
                        "content": caption,
                        "metadata": {
                            "source_file": filename,
                            "page_number": page_num,
                            "image_path": img_path,
                            "category": "Diagram / Technical Figure",
                            "doc_title": filename.replace(".pdf", "")
                        }
                    })
            except Exception as e:
                # Handle non-standard PDF image objects gracefully
                continue

        return extracted_diagrams
