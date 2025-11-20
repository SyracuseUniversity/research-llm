#hybrid_langchain_retriever.py
import chromadb
from chromadb.utils import embedding_functions
from runtime_settings import settings
import config_full as config

embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/e5-base-v2")

def _get_collection():
    client = chromadb.PersistentClient(
        path = config.CHROMA_DIR_FULL if settings.active_mode == "full" else config.CHROMA_DIR_ABSTRACTS
    )
    name = "papers_all" if settings.active_mode == "full" else "abstracts_all"
    return client.get_or_create_collection(name=name, embedding_function=embed)

def hybrid_search(query, k=12):
    col = _get_collection()
    r = col.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
    docs = r["documents"][0]
    metas = r["metadatas"][0]
    fused = [f"{m.get('title','Unknown')} ({m.get('year','?')}): {docs[i][:300]}" for i,m in enumerate(metas)]
    return {"chroma_ctx": fused, "metas": metas}
