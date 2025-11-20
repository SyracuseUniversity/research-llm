#chroma_ingest.py
import os, sqlite3, time
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import config_full as config

DB_PATH    = config.SQLITE_DB_FULL                  # ingestion only
CHROMA_DIR = config.CHROMA_DIR_FULL
BATCH_SIZE = 500

def safe_meta(val, default="N/A"):
    if val is None: return default
    if isinstance(val, (int, float, bool)): return val
    s = str(val).strip()
    return s if s else default

def split_into_halves(text: str):
    if not text: return []
    mid = len(text) // 2
    return [text[:mid], text[mid:]]

def build_doc(row):
    """
    row = (
      id, researcher_name, work_title, authors, info, doi, publication_date,
      file_name, summary, full_text
    )
    """
    rid, researcher, title, authors, info, doi, pub_date, file_name, summary, fulltext = row
    if not (title or summary or fulltext): return None
    text = (
        f"Researcher: {safe_meta(researcher, 'Unknown')}\n"
        f"Title: {safe_meta(title, 'Untitled')}\n"
        f"Authors: {safe_meta(authors)}\n"
        f"Info: {safe_meta(info)}\n"
        f"DOI: {safe_meta(doi)}\n"
        f"Publication Date: {safe_meta(pub_date)}\n"
        f"File: {safe_meta(file_name)}\n\n"
        f"Summary: {safe_meta(summary)}\n\n"
        f"Fulltext:\n{safe_meta(fulltext)}"
    )
    return rid, text, researcher, title, authors, doi, pub_date, file_name

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.SENTENCE_TFORMER
    )

    try:
        client.delete_collection("papers_all")
        print("🗑️ Old 'papers_all' collection removed.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="papers_all",
        embedding_function=embedder
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id, r.researcher_name, r.work_title, r.authors, r.info, r.doi, r.publication_date,
               w.file_name, w.summary, w.full_text
        FROM research_info r
        LEFT JOIN works w ON r.id = w.id
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"🔎 Found {len(rows)} combined rows to ingest (before filtering).")
    rows = [r for r in rows if (r[2] or r[8] or r[9])]
    print(f"📌 After filtering: {len(rows)} rows kept for ingestion.")

    total_start = time.time()
    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Ingesting", unit="batch"):
        batch = rows[start:start + BATCH_SIZE]
        docs, ids, metas = [], [], []

        for row in batch:
            built = build_doc(row)
            if not built: continue
            rid, text, researcher, title, authors, doi, pub_date, file_name = built
            chunks = split_into_halves(text)
            for i, chunk in enumerate(chunks, start=1):
                docs.append(chunk)
                ids.append(f"{rid}_part{i}")
                metas.append({
                    "id": safe_meta(rid),
                    "researcher": safe_meta(researcher, "Unknown"),
                    "title": safe_meta(title, "Untitled"),
                    "authors": safe_meta(authors),
                    "doi": safe_meta(doi),
                    "year": safe_meta(pub_date),
                    "file_name": safe_meta(file_name),
                    "chunk": i,
                })

        if docs:
            batch_start = time.time()
            collection.add(documents=docs, ids=ids, metadatas=metas)
            speed = len(docs) / (time.time() - batch_start + 1e-6)
            print(f"⚡ Batch {start//BATCH_SIZE+1}: {len(docs)} docs at {speed:.2f} docs/sec")

    print(f"✅ Ingestion complete — {len(rows)} works stored in `papers_all`.")
    print(f"⏱️ Total time: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
