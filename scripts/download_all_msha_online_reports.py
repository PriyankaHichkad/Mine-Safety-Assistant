import os
import re
import time
import urllib.request
from html.parser import HTMLParser

DATA_DIR = "./data/msha_reports"
os.makedirs(DATA_DIR, exist_ok=True)

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        if tag in ["script", "style"]:
            self.in_script = True

    def handle_endtag(self, tag):
        if tag in ["script", "style"]:
            self.in_script = False

    def handle_data(self, d):
        if not self.in_script and d.strip():
            self.fed.append(d.strip())

    def get_data(self):
        return " ".join(self.fed)

def strip_html(html_content: str) -> str:
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_data()
    except Exception:
        return re.sub(r'<[^>]+>', ' ', html_content)

def fetch_url(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception as err:
        print(f"   [Error fetching {url}: {err}]")
        return ""

def crawl_arlweb_index(year: int, category: str):
    """Crawls arlweb.msha.gov year indices (1995-2007) for Metal (FABM) and Coal (FABC)."""
    year_str = str(year)
    short_year = year_str[-2:]
    
    if year >= 2000:
        page_name = f"{category}{year_str}.asp"
    else:
        page_name = f"{category}{short_year}.HTM"
        
    index_url = f"https://arlweb.msha.gov/fatals/indices/{page_name}"
    print(f"\nScanning Index Page: {index_url}...")
    
    html = fetch_url(index_url)
    if not html:
        return

    # Find all report links like /FATALS/2007/FTL07m01.asp or /FATALS/2007/FAB07m01.asp
    report_links = re.findall(r'href=["\'](/FATALS/[^"\']+\.(?:asp|htm|html|pdf))["\']', html, re.IGNORECASE)
    report_links = list(set(report_links))
    print(f"  -> Found {len(report_links)} report links for {category} {year}!")

    for rel_link in report_links:
        full_url = f"https://arlweb.msha.gov{rel_link}"
        doc_id = rel_link.split("/")[-1].replace(".asp", "").replace(".htm", "").replace(".html", "").upper()
        filename = f"MSHA_ARLWEB_{year}_{doc_id}.txt"
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath):
            continue

        print(f"     Fetching report: {doc_id}...")
        report_html = fetch_url(full_url)
        if report_html:
            clean_text = strip_html(report_html)
            if len(clean_text) > 100:
                header = (
                    f"DOCUMENT_TITLE: MSHA Official Fatality Investigation Report {doc_id}\n"
                    f"DOC_ID: {doc_id}\n"
                    f"SOURCE_URL: {full_url}\n"
                    f"PUBLISHER: Mine Safety and Health Administration (MSHA) Official Report\n\n"
                )
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(header + clean_text)
                print(f"        [Saved {filename} ({len(clean_text)} chars)]")

def crawl_msha_modern_reports():
    """Crawls modern www.msha.gov fatality report pages."""
    base_search_url = "https://www.msha.gov/data-and-reports/fatality-reports/search?combine=&field_mine_category_tid=191&field_arep_fatal_date_value%5Bmin%5D%5Bdate%5D=2018-01-01&field_arep_fatal_date_value%5Bmax%5D%5Bdate%5D=2018-12-31&province=All&page="
    
    print("\nScanning Modern msha.gov Search Pages...")
    for page in range(0, 5):
        search_url = f"{base_search_url}{page}"
        print(f"  -> Scanning Page {page}: {search_url}")
        html = fetch_url(search_url)
        if not html:
            continue

        detail_links = re.findall(r'href=["\'](/data-reports/fatality-reports/[^"\']+)["\']', html, re.IGNORECASE)
        detail_links = list(set(detail_links))
        
        for rel_link in detail_links:
            full_url = f"https://www.msha.gov{rel_link}"
            clean_name = rel_link.strip("/").replace("/", "_").replace("-", "_")
            filename = f"MSHA_MODERN_{clean_name[:40]}.txt"
            filepath = os.path.join(DATA_DIR, filename)

            if os.path.exists(filepath):
                continue

            print(f"     Fetching modern report: {rel_link}...")
            report_html = fetch_url(full_url)
            if report_html:
                clean_text = strip_html(report_html)
                if len(clean_text) > 100:
                    header = (
                        f"DOCUMENT_TITLE: MSHA Modern Fatality Report\n"
                        f"SOURCE_URL: {full_url}\n"
                        f"PUBLISHER: Mine Safety and Health Administration (MSHA) Official Report\n\n"
                    )
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(header + clean_text)
                    print(f"        [Saved {filename}]")

def main():
    print("=== MSHA Official Web Fatality Scraper & Crawler ===")
    
    # Crawl Metal/Nonmetal (FABM) and Coal (FABC) indices from 1995 to 2007
    for yr in range(1995, 2008):
        crawl_arlweb_index(yr, "FABM")
        crawl_arlweb_index(yr, "FABC")

    # Crawl modern msha.gov reports
    crawl_msha_modern_reports()
    
    print("\n=== MSHA Web Scraper Crawl Complete! ===")

if __name__ == "__main__":
    main()
