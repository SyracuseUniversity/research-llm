#chroma_ingest_abstracts.py
"""
Builds a ChromaDB collection 'abstracts_all' from abstracts_only.db.
Each abstract becomes a vector document with DOI, title, source, and year metadata.
"""

import os, sqlite3, time
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import config_full as config

DB_PATH     = config.SQLITE_DB_ABSTRACTS       # C:\codes\t5-db\abstracts_only.db
CHROMA_DIR  = config.CHROMA_DIR_ABSTRACTS
COLLECTION_NAME = "abstracts_all"
BATCH_SIZE  = 500

def safe_str(v):
    if v is None: return ""
    return str(v).strip()

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.SENTENCE_TFORMER
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Old '{COLLECTION_NAME}' collection removed.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT doi, title, abstract, source, year
        FROM abstracts_only
        WHERE abstract IS NOT NULL AND abstract != ''
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"📚 Found {len(rows)} abstracts to ingest.")
    start_all = time.time()

    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Ingesting", unit="batch"):
        batch = rows[start:start+BATCH_SIZE]
        ids, docs, metas = [], [], []
        for idx, (doi, title, abstract, source, year) in enumerate(batch):
            if not abstract: continue
            ids.append(f"{start}_{idx}")
            docs.append(safe_str(abstract))
            metas.append({
                "doi": safe_str(doi),
                "title": safe_str(title),
                "source": safe_str(source),
                "year": safe_str(year)
            })
        if docs:
            t0 = time.time()
            collection.add(ids=ids, documents=docs, metadatas=metas)
            print(f"⚡ Batch {start//BATCH_SIZE+1}: {len(docs)} docs ({len(docs)/(time.time()-t0+1e-6):.2f}/sec)")

    print(f"✅ Ingest complete — {len(rows)} abstracts stored in '{COLLECTION_NAME}'.")
    print(f"⏱️ Total time: {time.time()-start_all:.2f}s")

if __name__ == "__main__":
    main()
