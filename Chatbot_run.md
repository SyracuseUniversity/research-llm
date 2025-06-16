---

# Running the Chatbot with Your QLoRA Adapter

Once you’ve fine-tuned LLaMA via QLoRA and have your adapter + tokenizer saved (e.g. in `models/llama_rag_qlora`), follow these steps to launch the chatbot:

## 1️⃣ Prerequisites

- **Python 3.8+**  
- **CUDA-enabled GPU** (optional but recommended)  
- **Dependencies installed**:
  ```bash
  pip install torch transformers peft bitsandbytes chromadb pandas numpy tqdm
  ```

**Artifacts in place:**

- Base LLaMA model at `C:\codes\llama32\Llama-3.2-1B-Instruct`  
- LoRA adapter + tokenizer at `models/llama_rag_qlora/`  
- ChromaDB index directory (e.g. `./chroma_storage`) populated via `migrate_to_chromadb.py`

## 2️⃣ Configure Environment (Optional)

If you prefer environment variables, set:

```bash
export BASE_MODEL="C:/codes/llama32/Llama-3.2-1B-Instruct"
export LORA_ADAPTER="models/llama_rag_qlora"
export CHROMA_DIR="./chroma_storage"
```

## 3️⃣ Launch the Chatbot

### Option A: Using `chatbot.py`
```bash
python chatbot.py   --base-model "C:/codes/llama32/Llama-3.2-1B-Instruct"   --adapter-dir "models/llama_rag_qlora"   --chroma-dir "./chroma_storage"
```

### Option B: Using `chatbot_standalone.py`
```bash
python chatbot_standalone.py   --model-dir "models/llama_rag_qlora"   --chroma-dir "./chroma_storage"
```

(If your scripts don’t accept flags, edit the top of `chatbot.py`/`chatbot_standalone.py` to point to your adapter and chroma paths.)

## 4️⃣ Interact

Once running, you’ll see a prompt like:
```makefile
You: <your question>
Assistant: <model’s answer>
```
