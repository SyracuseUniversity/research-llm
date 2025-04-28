# train_llama_only.py

import pandas as pd
import torch
import sqlite3
import os
from llama_model import fine_tune_llama_on_papers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def clear_memory():
    """Clear GPU and CPU memory."""
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("Cleared memory and cache.")

def fine_tune_only_llama():
    """ONLY Step 4: Fine-tune LLaMA on Syracuse metadata."""

    print("STEP 4: Fine-tuning LLaMA on Syracuse metadata...")

    # Load metadata training file
    metadata_path = r"C:\codes\t5-db\processed_metadata_training.pkl"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata training file not found at: {metadata_path}")

    with open(metadata_path, "rb") as f:
        metadata_df = pd.read_pickle(f)

    print(f"Loaded metadata training data with {len(metadata_df)} samples.")

    # Fine-tune
    fine_tune_llama_on_papers(metadata_df)
    print("✅ LLaMA fine-tuning completed.")

if __name__ == "__main__":
    fine_tune_only_llama()
