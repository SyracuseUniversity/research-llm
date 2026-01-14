# #hybrid_langchain_retriever.py
# import chromadb
# from chromadb.utils import embedding_functions
# from runtime_settings import settings
# import config_full as config

# embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/e5-base-v2")

# def _get_collection():
#     client = chromadb.PersistentClient(
#         path = config.CHROMA_DIR_FULL if settings.active_mode == "full" else config.CHROMA_DIR_ABSTRACTS
#     )
#     name = "papers_all" if settings.active_mode == "full" else "abstracts_all"
#     return client.get_or_create_collection(name=name, embedding_function=embed)

# def hybrid_search(query, k=12):
#     col = _get_collection()
#     r = col.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
#     docs = r["documents"][0]
#     metas = r["metadatas"][0]
#     fused = [f"{m.get('title','Unknown')} ({m.get('year','?')}): {docs[i][:300]}" for i,m in enumerate(metas)]
#     return {"chroma_ctx": fused, "metas": metas}


# hybrid_langchain_retriever.py
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from runtime_settings import settings
import config_full as config

import psutil
import torch


# Embedding model (E5 base)
EMBED_MODEL = "intfloat/e5-base-v2"
EMBED_DIM = 768  # E5 base embedding dimension
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# MiniLM cross encoder reranker
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL)


# Dynamic candidate count based on hardware
def compute_dynamic_k(total_vectors, embedding_dim=EMBED_DIM, dtype_bytes=4, use_gpu=False):
    if total_vectors <= 0:
        return 0

    mem = psutil.virtual_memory()
    available_ram = mem.available

    safety = 0.8
    usable_ram = available_ram * safety

    bytes_per_vector = embedding_dim * dtype_bytes
    max_vectors_ram = usable_ram // bytes_per_vector

    if use_gpu and torch.cuda.is_available():
        free_vram, _ = torch.cuda.mem_get_info()
        usable_vram = free_vram * safety
        max_vectors_gpu = usable_vram // bytes_per_vector
        max_vectors = min(max_vectors_ram, max_vectors_gpu)
    else:
        max_vectors = max_vectors_ram

    return int(min(total_vectors, max_vectors))


# Select DB
def get_chroma():
    path = (
        config.CHROMA_DIR_FULL
        if settings.active_mode == "full"
        else config.CHROMA_DIR_ABSTRACTS
    )
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name="papers_all",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# Standardize metadata
def normalize_metadata(meta):
    meta = meta or {}
    return {
        "title": meta.get("title", "Unknown"),
        "year": meta.get("year", "?"),
        "authors": meta.get("authors", ""),
        "researcher": meta.get("researcher_name", meta.get("researcher", "")),
        "snippet": meta.get("snippet", meta.get("abstract", meta.get("text", ""))),
        "text": meta.get("text", meta.get("fulltext", meta.get("snippet", ""))),
    }


# Vector search
def vector_search(col, query, k):
    if k <= 0:
        return []

    r = col.query(
        query_texts=[query],
        n_results=int(k),
        include=["ids", "metadatas", "documents", "distances"],
    )

    ids = r["ids"][0]
    metas = r["metadatas"][0]
    docs = r["documents"][0]
    dists = r["distances"][0]

    out = []
    for doc_id, meta, doc, dist in zip(ids, metas, docs, dists):
        m = normalize_metadata(meta)
        out.append(
            {
                "id": doc_id,
                "title": m["title"],
                "year": m["year"],
                "authors": m["authors"],
                "researcher": m["researcher"],
                "snippet": m["snippet"],
                "text": m["text"],
                "similarity": 1 - dist,
            }
        )
    return out


# Rerank with MiniLM
def rerank_passages(query, items, top_k=None):
    if not items:
        return []

    pairs = [[query, it["text"]] for it in items]
    scores = reranker.predict(pairs)

    ranked = sorted(
        [{**it, "similarity": float(scores[i])} for i, it in enumerate(items)],
        key=lambda x: x["similarity"],
        reverse=True,
    )

    if top_k is None:
        return ranked
    return ranked[:top_k]


def select_top_papers(items, max_papers=20, min_score=0.6):
    if not items:
        return []

    by_pid = {}

    for it in items:
        score = it.get("similarity", 0.0)
        if score < min_score:
            continue
        pid = it.get("id") or f"{it.get('title', '')}|{it.get('year', '')}"
        by_pid.setdefault(pid, []).append(it)

    if not by_pid:
        return items[:max_papers]

    for pid in by_pid:
        by_pid[pid].sort(key=lambda x: x["similarity"], reverse=True)

    ordered_pids = sorted(
        by_pid.keys(),
        key=lambda pid: by_pid[pid][0]["similarity"],
        reverse=True,
    )

    selected = []
    for pid in ordered_pids[:max_papers]:
        selected.extend(by_pid[pid])

    return selected


# Main hybrid retrieval
def hybrid_search(query, top_k=15):
    col = get_chroma()

    total_vectors = col.count()
    k_dynamic = compute_dynamic_k(total_vectors, use_gpu=False)
    if k_dynamic <= 0:
        k_dynamic = max(top_k, 20)

    candidates = vector_search(col, query, k=k_dynamic)
    if not candidates:
        return []

    ranked = rerank_passages(query, candidates, top_k=None)

    selected = select_top_papers(ranked, max_papers=20, min_score=0.6)
    if not selected:
        selected = ranked[:top_k]

    for f in selected:
        if len(f["text"]) > 2000:
            f["text"] = f["text"][:2000] + "..."
        if len(f["snippet"]) > 500:
            f["snippet"] = f["snippet"][:500] + "..."

    return selected
