# Research Paper Summarization and Fine-Tuning Pipeline

This repository contains a workflow for:
1. **Extracting text from PDFs** and storing them in a local SQLite database,
2. **Summarizing PDFs** using a T5 model,
3. **Fine-tuning the T5 model** with the summarized text,
4. **Fine-tuning a LLaMA model** for research Q&A, and
5. **Interacting with the fine-tuned LLaMA model** via a chatbot interface.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Dependencies and Installation](#dependencies-and-installation)  
3. [Usage](#usage)  
4. [Scripts Overview](#scripts-overview)  
   - [chatbot.py](#chatbotpy)  
   - [data_pre.py](#data_prepy)  
   - [database_handler.py](#database_handlerpy)  
   - [llama_model.py](#llama_modelpy)  
   - [main.py](#mainpy)  
   - [pdf_pre.py](#pdf_prepy)  
   - [t5_model.py](#t5_modelpy)  
   - [train_llama.py](#train_llamapy)  
5. [Notes and Tips](#notes-and-tips)

---

## Project Structure

├── chatbot.py

├── data_pre.py

├── database_handler.py

├── llama_model.py

├── main.py

├── pdf_pre.py

├── t5_model.py

├── train_llama.py

├── researchers.db            # SQLite database (auto-generated if not present)

├── download_pdfs/           # Folder containing your PDFs

│   ├── paper1.pdf

│   ├── paper2.pdf

│   └── ...

├── fine_tuned_t5/           # Directory for fine-tuned T5 model (created by main.py)

├── fine_tuned_llama/        # Directory for fine-tuned LLaMA model (created by train_llama.py)

├── Llama-3.2-1B-Instruct/   # Base LLaMA model directory (assumes local copy)

└── processed_training_data.pkl # Pickled training data for LLaMA (created by train_llama.py)


---

## Dependencies and Installation

**Python 3.7+ recommended.**

Create and activate a virtual environment (optional but recommended):

\`\`\`bash
python -m venv venv

# On Linux or Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
\`\`\`

Install required packages:

\`\`\`bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 \
transformers datasets PyPDF2==3.0.1 pandas pysqlite3
\`\`\`

Ensure you have your LLaMA base model folder (\`Llama-3.2-1B-Instruct/\` in the example) placed within the same directory. You may need to adjust the folder name or path in \`llama_model.py\`.

---

## Usage

1. **Place your PDFs** in the \`download_pdfs/\` folder.

2. **Run the main pipeline**:
    \`\`\`bash
    python main.py
    \`\`\`
    This script performs:
    - **STEP 1**: Reads all PDFs in \`download_pdfs/\`, extracts text, and populates \`researchers.db\`.
    - **STEP 2**: Summarizes unsummarized papers using the T5 model (the base \`t5-small\` by default).
    - **STEP 3**: Fine-tunes the T5 model on the newly generated summaries, saving the model to the \`fine_tuned_t5/\` folder.

3. (Optional) **Fine-tune LLaMA** on your summarized data by running:
    \`\`\`bash
    python train_llama.py
    \`\`\`
    This script:
    - Loads summarized data from \`researchers.db\`.
    - Saves a pickle file (\`processed_training_data.pkl\`).
    - Fine-tunes the LLaMA model, saving it to \`fine_tuned_llama/\`.

4. **Chat with the fine-tuned LLaMA** by running:
    \`\`\`bash
    python chatbot.py
    \`\`\`
    Type your questions into the console prompt, and the chatbot will generate answers using the fine-tuned LLaMA model.

---

## Scripts Overview

### \`chatbot.py\`
- Launches a command-line chatbot.
- Checks if the user’s input is a greeting; if so, it responds with a friendly message.
- Otherwise, it queries \`chatbot_answer\` from \`llama_model.py\` for a response.

### \`data_pre.py\`
- Contains a helper function \`preprocess_text_for_t5\` to tokenize and prepare text for T5 summarization.

### \`database_handler.py\`
- Manages an SQLite database (\`researchers.db\`).
- Provides functions to:
  - **Create tables** (\`setup_database\`)
  - **Insert or update records** (\`insert_work\`, \`update_summary\`)
  - **Fetch records** (\`fetch_unsummarized_works\`)
  - **Remove duplicates** (\`remove_duplicates\`)
  - **Count entries** (\`count_entries_in_table\`)
  - **Identify missing files** (\`check_missing_files_in_db\`)

### \`llama_model.py\`
- Loads a base LLaMA model or a fine-tuned LLaMA model if one exists.
- Implements \`chatbot_answer\` for generating responses to user queries.
- Implements \`fine_tune_llama_on_papers\` to fine-tune LLaMA using the text–summary pairs.

### \`main.py\`
- High-level pipeline:
  - \`populate_database_from_pdfs()\`: Reads PDFs from \`download_pdfs/\`, extracts text, and populates the database.
  - \`generate_summaries_for_database()\`: Summarizes unsummarized papers using T5.
  - \`fine_tune_model_on_summaries()\`: Fine-tunes T5 on the summarized data.

### \`pdf_pre.py\`
- Contains functions for PDF text extraction (\`extract_text_from_pdf\`) using PyPDF2.
- Cleans the extracted text to remove unwanted characters.

### \`t5_model.py\`
- Loads the base \`t5-small\` model for summarization.
- Implements \`summarize_text\` for quick one-off summarization.
- Implements \`fine_tune_t5_on_papers\` to further fine-tune T5 on your dataset.

### \`train_llama.py\`
- Loads summarized data from the database.
- Saves that data as a pickle file for quick reloads.
- Calls \`fine_tune_llama_on_papers\` to fine-tune the LLaMA model on the T5-generated summaries.

---

## Notes and Tips

- Ensure that your PDFs are stored in the \`download_pdfs/\` folder before running the pipeline.
- Check the output directories for the fine-tuned models (\`fine_tuned_t5/\` and \`fine_tuned_llama/\`).
- LLaMA fine-tuning may take a long time, depending on the dataset size and model configuration.
