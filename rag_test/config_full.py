#config_full.py
import os

# --- Chroma Database dirs ---
CHROMA_DIR_FULL = r"C:\codes\t5-db\chroma_store_full"
CHROMA_DIR_ABSTRACTS = r"C:\codes\t5-db\chroma_store_abstracts"

# --- Neo4j ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "OSPOlol@1234"
NEO4J_DB = "syr-rag"   # used for both full + abstracts for simplicity

# --- LLMs ---
LLAMA_1B = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
LLAMA_3B = r"C:\codes\llama32\Llama-3.2-3B-Instruct"
