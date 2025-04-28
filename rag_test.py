# #PARTIAL
# # rag_test_multiturn.py

# import torch
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
# import chromadb

# # -------------------
# # Configuration
# # -------------------
# LOAD_IN_8BIT = True  # <<--- Set this to False if you want full precision
# CHROMA_DIR = r"D:\OSPO\ChromaDB"
# LLAMA_MODEL_PATH = r"D:\OSPO\llama321b"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # -------------------
# # Load components
# # -------------------

# # 1. Load embedding model
# embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# # 2. Connect to ChromaDB
# chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# vectorstore = Chroma(
#     client=chroma_client,
#     collection_name="research_info_collection",
#     embedding_function=embedder,
# )

# # 3. Load base LLaMA model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)

# if LOAD_IN_8BIT:
#     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#     llama_model = AutoModelForCausalLM.from_pretrained(
#         LLAMA_MODEL_PATH,
#         quantization_config=quantization_config,
#         device_map="auto",
#     )
# else:
#     llama_model = AutoModelForCausalLM.from_pretrained(
#         LLAMA_MODEL_PATH
#     ).to(device)

# generator_pipeline = pipeline(
#     "text-generation",
#     model=llama_model,
#     tokenizer=llama_tokenizer,
#     max_new_tokens=300,
#     temperature=0.7,
#     top_p=0.9,
#     repetition_penalty=1.1,
#     #device=0 if torch.cuda.is_available() else -1
# )

# llm = HuggingFacePipeline(pipeline=generator_pipeline)

# # -------------------
# # Multi-turn chat loop
# # -------------------

# print("\nSyracuse RAG Chatbot (Multi-Turn, With Thinking) is ready.")
# print("Type your research questions below (type 'exit' to quit).\n")

# chat_history = []

# while True:
#     query = input("You: ").strip()
#     if query.lower() in {"exit", "quit"}:
#         print("\nExiting chatbot. Goodbye.")
#         break

#     docs = vectorstore.similarity_search(query, k=3)

#     if not docs:
#         print("No relevant documents found.\n")
#         continue

#     context = "\n\n".join(doc.page_content for doc in docs)

#     final_prompt = (
#         "You are a Syracuse University research assistant.\n"
#         "You must show your full thinking process step-by-step when answering research questions.\n"
#         "Use the retrieved information below carefully.\n\n"
#         f"Retrieved Context:\n{context}\n\n"
#         f"User Question: {query}\n"
#         "Answer (show your thought process, and final answer clearly):"
#     )

#     response = llm.invoke(final_prompt)

#     if isinstance(response, list):
#         response = response[0]["generated_text"]

#     if "Answer:" in response:
#         answer = response.split("Answer:")[-1].strip()
#     else:
#         answer = response.strip()

#     print("\nChatbot Response (with full thinking):\n")
#     print(answer)
#     print("\n" + "="*80 + "\n")

#     chat_history.append({"question": query, "answer": answer})


import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import chromadb

# -------------------
# Configuration
# -------------------
LOAD_IN_8BIT = True
CHROMA_DIR = r"D:\OSPO\ChromaDB"
LLAMA_MODEL_PATH = r"D:\OSPO\llama321b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------
# Load components
# -------------------

# 1. Load embedding model
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 2. Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="research_info_collection",
    embedding_function=embedder,
)

# 3. Load base LLaMA model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True, local_files_only=True)

if LOAD_IN_8BIT:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
else:
    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

generator_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=generator_pipeline)

# -------------------
# Multi-turn chat loop
# -------------------

print("\nSyracuse RAG Chatbot (Multi-Turn) is ready.")
print("Type your research questions below (type 'exit' to quit).\n")

chat_history = []

while True:
    query = input("You: ").strip()
    if query.lower() in {"exit", "quit"}:
        print("\nExiting chatbot. Goodbye.")
        break

    docs = vectorstore.similarity_search(query, k=5)

    if not docs:
        print("No relevant documents found.\n")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = (
        "You are a Syracuse University research assistant.\n"
        "You must use ONLY the retrieved information below.\n"
        "If the information is not found, you must clearly say: 'The retrieved context does not contain this information.'\n"
        "Do not guess. Do not use outside knowledge.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Question: {query}\n"
        "Answer (show your thought process, then conclude clearly):"
    )

    response = llm.invoke(final_prompt)

    if isinstance(response, list):
        response = response[0]["generated_text"]

    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()

    print("\nChatbot Response (with full thinking):\n")
    print(answer)
    print("\n" + "="*80 + "\n")

    chat_history.append({"question": query, "answer": answer})
