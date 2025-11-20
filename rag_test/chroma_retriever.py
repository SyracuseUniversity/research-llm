#chroma_retriever.py
import chromadb, re
from chromadb.utils import embedding_functions
import config_full as config
from runtime_settings import settings

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/e5-base-v2"
)

def _year(meta):
    if not meta: return 0
    s = str(meta.get("year") or meta.get("publication_year") or "")
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else 0

def get_collection():
    mode = settings.active_mode or config.ACTIVE_MODE
    if mode == "abstracts":
        client = chromadb.PersistentClient(path=config.CHROMA_DIR_ABSTRACTS)
        name = config.COLLECTION_ABSTRACTS
    else:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR_FULL)
        name = config.COLLECTION_FULL
    return client.get_or_create_collection(name=name, embedding_function=embedder)

def query_chroma(question: str, k: int = 8, threshold: float = None):
    col = get_collection()
    r = col.query(query_texts=[question], n_results=k, include=["documents", "metadatas", "distances"])
    items = list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))
    # Sort by year DESC regardless of distance; ignore month/day by design
    items.sort(key=lambda x: _year(x[1]), reverse=True)
    return [(doc, meta) for (doc, meta, dist) in items if threshold is None or dist < threshold]
