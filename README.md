# Pipeline Overview

This project implements a full RAG‑style research assistant:

1. **`train_pipeline.py`** — end‑to‑end training:  
   - PDF ingestion  
   - SQLite staging  
   - T5 summarization & fine‑tuning  
   - “Lite” dataset creation  
   - LLaMA fine‑tuning  

2. **`rag_pipeline.py`** — interactive RAG chatbot:  
   - ChromaDB vector store  
   - LangChain RetrieverQA  
   - Fine‑tuned LLaMA generation  

3. **`test_rag.py`** — automated test suite for your RAG chatbot.

All scripts assume your code lives under `C:\codes\…`.

---

## Folder Structure
pipeline_project/ 
├── train_pipeline.py

├── rag_pipeline.py

├── test_rag.py

├── migrate_sqlite_to_chromadb.py

├── pdfs.py

├── pdf_pre.py

├── model.py

├── llama_model.py

├── database_handler.py

├── data_pre.py

└── (optional helper scripts)


- **`train_pipeline.py`**  
  Orchestrates data ingestion, summarization, and model fine‑tuning in one shot.

- **`rag_pipeline.py`**  
  Loads ChromaDB + LangChain + LLaMA to serve an interactive chatbot.

- **`test_rag.py`**  
  Runs a list of test questions through your RAG chain and logs outputs.

- **`migrate_sqlite_to_chromadb.py`**  
  One‑time migration of all SQLite data into ChromaDB.

- **`pdfs.py`** / **`pdf_pre.py`**  
  Download, extract, and clean PDF text.

- **`model.py`**  
  T5 summarization & fine‑tuning utilities.

- **`llama_model.py`**  
  LLaMA fine‑tuning utilities.

- **`database_handler.py`**  
  SQLite schema & CRUD helpers.

- **`data_pre.py`**  
  Text‑to‑T5 preprocessing helper.

---

# Detailed File and Function Documentation

## 1. train_pipeline.py

### Purpose  
Runs the full data → model training loop:

1. **Download PDFs** (from your merged CSV)  
2. **Extract & ingest** into SQLite  
3. **Summarize** with T5 and update DB  
4. **Fine‑tune T5** on `(full_text → summary)`  
5. **Create lite DB/pickle/CSV**  
6. **Fine‑tune LLaMA** on `(input_text → target_text)`

### Key Functions  
- **`download_pdfs()`**  
- **`process_pdfs_into_sqlite()`**  
- **`generate_summaries_and_finetune_t5()`**  
- **`create_lite_and_finetune_llama()`**

Each step is checkpoint‑aware and resumes from the latest checkpoint.

---

## 2. rag_pipeline.py

### Purpose  
Serves an interactive Retrieval‑Augmented Generation chatbot:

1. Loads ChromaDB (persisted vector store)  
2. Uses Sentence‑Transformers embeddings  
3. Instantiates a LangChain `RetrievalQA` chain  
4. Wraps your fine‑tuned LLaMA in a HuggingFacePipeline  
5. Provides a REPL chat loop

### Key Sections  
- **Configuration** (paths, model names, device)  
- **Retriever instantiation**  
- **LLM loading & pipeline**  
- **`RetrievalQA.from_chain_type(...)`**  
- **`chat()`** loop

---

## 3. test_rag.py

### Purpose  
Executes a predefined list of questions against your RAG pipeline and logs results:

- **`TEST_QUESTIONS`** array  
- **`log_entry()`** writes to CSV  
- **`if __name__ == "__main__":`** iterates, runs `qa.run(...)`, logs

---

## 4. migrate_sqlite_to_chromadb.py

### Purpose  
One‑time migration of your SQLite tables into ChromaDB:

- Fetches `works` and `research_info`  
- Chunks long texts with `RecursiveCharacterTextSplitter`  
- Embeds with Sentence‑Transformers (`all-MiniLM-L6-v2`)  
- Adds documents to Chroma collection and persists

---

## 5. pdfs.py & pdf_pre.py

- **`pdfs.py`**: Downloads PDFs from a CSV.  
- **`pdf_pre.py`**:  
  - `extract_text_from_pdf(file_path)`  
  - `clean_text(text)`  
  - `extract_research_info_from_pdf(file_path)`

---

## 6. model.py

### Purpose  
T5 summarization & fine‑tuning helpers:

- `summarize_text(text, idx=None, total=None)`  
- `fine_tune_t5_on_papers(dataset, output_dir)`

Supports resuming from the latest checkpoint.

---

## 7. llama_model.py

### Purpose  
LLaMA fine‑tuning helpers:

- `fine_tune_llama_on_papers(dataset, output_dir)`  
  - Masks prompt tokens, computes loss only on summary tokens  
  - Resumes from checkpoint

- `clear_memory()`

---

## 8. database_handler.py

CRUD operations for SQLite:

- `setup_database()`, `setup_research_info_table()`  
- `insert_work(...)`, `remove_duplicates()`, `fetch_unsummarized_works()`  
- `update_summary(work_id, summary)`  
- `insert_research_info(...)`, `fetch_research_info()`  
- `count_entries_in_table()`, `check_missing_files_in_db(...)`

---

## 9. data_pre.py

- `preprocess_text_for_t5(text, model_name="t5-small")`

---

# Dependency Diagram


- **`train_pipeline.py`**  
  Orchestrates data ingestion, summarization, and model fine‑tuning in one shot.

- **`rag_pipeline.py`**  
  Loads ChromaDB + LangChain + LLaMA to serve an interactive chatbot.

- **`test_rag.py`**  
  Runs a list of test questions through your RAG chain and logs outputs.

- **`migrate_sqlite_to_chromadb.py`**  
  One‑time migration of all SQLite data into ChromaDB.

- **`pdfs.py`** / **`pdf_pre.py`**  
  Download, extract, and clean PDF text.

- **`model.py`**  
  T5 summarization & fine‑tuning utilities.

- **`llama_model.py`**  
  LLaMA fine‑tuning utilities.

- **`database_handler.py`**  
  SQLite schema & CRUD helpers.

- **`data_pre.py`**  
  Text‑to‑T5 preprocessing helper.

---

# Dependency & Pipeline Flowchart

```mermaid
flowchart LR
    subgraph Training Pipeline
      A[train_pipeline.py]
    end

    subgraph RAG Service
      B[rag_pipeline.py]
      C[test_rag.py]
    end

    subgraph Helpers
      D[migrate_sqlite_to_chromadb.py]
      E[pdfs.py] & F[pdf_pre.py]
      G[model.py] & H[llama_model.py]
      I[database_handler.py] & J[data_pre.py]
    end

    A --> I
    A --> E
    A --> F
    A --> G
    A --> H

    B --> D
    B --> G
    B --> H

    C --> B

