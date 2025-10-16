import chromadb
from chromadb.utils import embedding_functions
import config_full as config

client = chromadb.PersistentClient(path=config.CHROMA_DIR)

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/e5-base-v2"
)

collection = client.get_or_create_collection(
    name="papers_all",
    embedding_function=embedder
)

def query_chroma(question: str, k: int = 5, threshold: float = 0.4):
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [(doc, meta) for doc, meta, dist in zip(docs, metas, dists) if dist < threshold]
