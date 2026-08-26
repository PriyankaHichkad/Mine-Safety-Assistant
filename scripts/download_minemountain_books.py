import os
import re
import json
import urllib.request
from bs4 import BeautifulSoup

def download_minemountain_library():
    url = "https://minemountain.in/mining-e-library"
    save_dir = "./data/pdf_books"
    catalog_path = "./data/minemountain_catalog.json"
    
    os.makedirs(save_dir, exist_ok=True)
    print("=== MineMountain PDF E-Library Downloader ===")
    print(f"Fetching catalog from {url}...")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    )
    
    try:
        html = urllib.request.urlopen(req).read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching website: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "example"}) or soup.find("table")
    
    if not table:
        print("Table not found on page.")
        return

    rows = table.find_all("tr")
    print(f"Found {len(rows)} table entries. Processing downloadable PDFs...")
    
    catalog = []
    download_count = 0
    
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
            
        book_title = cols[1].get_text(strip=True)
        author = cols[2].get_text(strip=True)
        
        pdf_link_tag = cols[3].find("a", href=True)
        if not pdf_link_tag:
            continue
            
        pdf_url = pdf_link_tag["href"]
        if not pdf_url.endswith(".pdf"):
            continue
            
        # Clean filename
        safe_title = re.sub(r"[^\w\-]", "_", book_title)[:50]
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(save_dir, filename)
        
        catalog.append({
            "book_title": book_title,
            "author": author,
            "filename": filename,
            "pdf_url": pdf_url
        })
        
        if not os.path.exists(filepath):
            print(f"Downloading [{download_count + 1}]: '{book_title}' by {author}...")
            try:
                urllib.request.urlretrieve(pdf_url, filepath)
                download_count += 1
            except Exception as err:
                print(f"  --> Failed to download {pdf_url}: {err}")
        else:
            print(f"Already exists: {filename}")

    # Save catalog mapping
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    print(f"\nCompleted! Catalog saved to {catalog_path}.")
    print(f"Total PDFs available in catalog: {len(catalog)} | Downloaded in this run: {download_count}")

if __name__ == "__main__":
    download_minemountain_library()
