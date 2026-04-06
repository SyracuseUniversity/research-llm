#test_chroma.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import config_full as config
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
embed = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2",
                              model_kwargs={"device": device})

client = chromadb.PersistentClient(path=config.CHROMA_DIR_FULL)

print("Collections present:")
print(client.list_collections())

# TRY EVERY COLLECTION AND SEE WHICH ONE HAS DATA
for c in client.list_collections():
    col = Chroma(client=client, collection_name=c.name, embedding_function=embed)
    try:
        docs = col.similarity_search("gravitational waves", k=5)
        print(f"\nCOLLECTION: {c.name} → {len(docs)} results")
    except Exception as e:
        print(f"ERROR on {c.name}: {e}")