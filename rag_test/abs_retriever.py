"""
api_abstract_retriever.py – Build abstracts_only.db via official APIs (no scraping)

Sources used (in order of preference):
  1. Crossref  – https://api.crossref.org/works/{doi}
  2. OpenAlex  – https://api.openalex.org/works/doi:{doi}
  3. arXiv     – http://export.arxiv.org/api/query?search_query=id:{arxiv_id}
"""

import os, re, time, html, json, requests, sqlite3
import pandas as pd
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATHS = [
    r"C:\Users\arapte\Downloads\Application\author_works.csv",
    r"C:\Users\arapte\Downloads\Application\cleaned_author_works.csv",
    r"C:\Users\arapte\Downloads\Application\filtered_author_works.csv",
    r"C:\Users\arapte\Downloads\Application\merged_cleaned.csv",
    r"C:\Users\arapte\Downloads\Application\syracuse_university_orcid_data.csv",
]
DB_OUT = r"C:\codes\t5-db\abstracts_only.db"
EMAIL_FOR_UNPAYWALL = "your_email@domain.com"   # required by Crossref policy

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
ARXIV_RE = re.compile(r"10\.48550/arXiv\.[\w\-\.]+", re.I)

def clean_abs(a: str) -> str:
    if not a: return ""
    a = html.unescape(a)
    a = re.sub(r"<[^>]+>", " ", a)
    a = re.sub(r"\s+", " ", a)
    return a.strip()

def init_db():
    conn = sqlite3.connect(DB_OUT)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS abstracts_only (
        doi TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        source TEXT,
        year TEXT,
        retrieved_at TEXT
    );
    """)
    conn.commit(); conn.close()

def safe_commit(cur, data):
    cur.execute("""
        INSERT OR IGNORE INTO abstracts_only
        (doi,title,abstract,source,year,retrieved_at)
        VALUES (?,?,?,?,?,datetime('now'))
    """, data)

# ─────────────────────────────────────────────────────────────────────────────
# API FETCHERS
# ─────────────────────────────────────────────────────────────────────────────
def get_crossref(doi):
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}",
                         headers={"User-Agent": f"mailto:{EMAIL_FOR_UNPAYWALL}"}, timeout=15)
        if r.status_code != 200: return None
        msg = r.json().get("message", {})
        abs_text = clean_abs(msg.get("abstract"))
        if abs_text:
            title = (msg.get("title") or [""])[0]
            year = str(msg.get("issued", {}).get("date-parts", [[None]])[0][0] or "")
            return {"title": title, "abstract": abs_text, "source": "crossref", "year": year}
    except Exception: pass
    return None

def get_openalex(doi):
    try:
        r = requests.get(f"https://api.openalex.org/works/doi:{doi}", timeout=15)
        if r.status_code != 200: return None
        msg = r.json()
        abs_text = msg.get("abstract")
        if abs_text:
            title = msg.get("display_name", "")
            year = str(msg.get("publication_year") or "")
            return {"title": title, "abstract": abs_text.strip(), "source": "openalex", "year": year}
    except Exception: pass
    return None

def get_arxiv(doi):
    try:
        arxivid = doi.split("arXiv.")[-1]
        url = f"http://export.arxiv.org/api/query?search_query=id:{arxivid}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return None
        root = ET.fromstring(r.text)
        ns = {'atom':'http://www.w3.org/2005/Atom'}
        summary = root.find('.//atom:summary', ns)
        title = root.find('.//atom:title', ns)
        year_el = root.find('.//atom:published', ns)
        if summary is not None:
            year = year_el.text[:4] if year_el is not None else ""
            return {"title": title.text.strip(), "abstract": summary.text.strip(),
                    "source": "arxiv", "year": year}
    except Exception: pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_dois(csv_paths):
    all_dois = set()
    for path in csv_paths:
        if not os.path.isfile(path): continue
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception:
            continue
        for col in df.columns:
            for cell in df[col].dropna().astype(str):
                for m in DOI_RE.findall(cell):
                    all_dois.add(m.strip())
    print(f"🧩 Extracted {len(all_dois)} unique DOIs from CSVs.")
    return sorted(all_dois)

def main():
    init_db()
    dois = extract_dois(CSV_PATHS)
    conn = sqlite3.connect(DB_OUT)
    cur = conn.cursor()
    found = 0

    for i, doi in enumerate(dois, 1):
        try:
            rec = None
            if doi.startswith("10.48550/arXiv"):
                rec = get_arxiv(doi)
            if not rec:
                rec = get_crossref(doi)
            if not rec:
                rec = get_openalex(doi)
            if rec and rec["abstract"]:
                safe_commit(cur, (doi, rec["title"], rec["abstract"], rec["source"], rec["year"]))
                conn.commit(); found += 1
        except Exception as e:
            print(f"Error for {doi}: {e}")

        if i % 50 == 0:
            print(f"Processed {i}/{len(dois)} DOIs  —  {found} abstracts saved.")
            time.sleep(1)  # polite delay

    conn.close()
    print(f"✅ Done. {found}/{len(dois)} abstracts saved to {DB_OUT}")

if __name__ == "__main__":
    main()
