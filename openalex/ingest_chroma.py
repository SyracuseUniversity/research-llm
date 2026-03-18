"""
ingest_chroma.py — Embed and index chunks into ChromaDB.

Reads:   data/raw/normalized_works.jsonl
Writes:  data/chroma_db/  (ChromaDB persistent store)

Standalone:
    python ingest_chroma.py
    python ingest_chroma.py --incremental
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from chunker import chunk_work

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000


def _safe(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, (int, float, bool)):
        return val
    return str(val).strip()


def run(rebuild: bool = True) -> Dict[str, Any]:
    works_file = Path(config.RAW_DIR) / "normalized_works.jsonl"
    if not works_file.exists():
        logger.error("No normalized_works.jsonl — run normalize first")
        return {"works": 0, "chunks": 0}

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        logger.error("chromadb not installed — run: pip install chromadb")
        return {"error": "chromadb not installed"}

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL,
        device=config.EMBED_DEVICE,
    )

    if rebuild:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
            logger.info("Deleted existing collection '%s'", config.CHROMA_COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    batch_ids:   List[str]  = []
    batch_docs:  List[str]  = []
    batch_metas: List[Dict] = []
    chunk_type_counts: Dict[str, int] = {}
    total_chunks = total_works = 0
    t0 = time.time()

    with open(works_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            work   = json.loads(line)
            chunks = chunk_work(work)
            if not chunks:
                continue

            total_works += 1
            for chunk in chunks:
                meta = {k: _safe(v) for k, v in chunk["metadata"].items()}
                meta["work_id"] = chunk["work_id"]

                batch_ids.append(chunk["chunk_id"])
                batch_docs.append(chunk["embed_text"])
                batch_metas.append(meta)
                total_chunks += 1

                ct = chunk["chunk_type"]
                chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

            if len(batch_ids) >= BATCH_SIZE:
                collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                elapsed = time.time() - t0
                logger.info(
                    "Upserted %d chunks | total: %d from %d works | %.0f chunks/s",
                    len(batch_ids), total_chunks, total_works,
                    total_chunks / max(0.1, elapsed),
                )
                batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    if batch_ids:
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    elapsed = time.time() - t0
    logger.info(
        "Chroma done: %d chunks from %d works in %.1fs (%.0f chunks/s)",
        total_chunks, total_works, elapsed, total_chunks / max(0.1, elapsed),
    )
    logger.info("Chunk types: %s", chunk_type_counts)

    return {
        "works": total_works,
        "chunks": total_chunks,
        "elapsed_s": round(elapsed, 1),
        **chunk_type_counts,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    print(run(rebuild=not args.incremental))