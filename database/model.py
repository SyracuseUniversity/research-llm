"""
model.py  –  T5 summarization and fine-tuning utilities.
"""

import os
import gc
import torch
import random
import numpy as np
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset

MODEL_NAME = "t5-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t5_model = None
t5_tokenizer = None

SUMMARIZE_MAX_INPUT = 512
SUMMARIZE_MAX_TARGET = 150
TRUNCATE_TEXT_CHARS = 1000
SEED = 42


def load_t5_model():
    """
    Lazy-load the T5 model and tokenizer. Returns (model, tokenizer, device).
    """
    global t5_model, t5_tokenizer

    if t5_model is not None and t5_tokenizer is not None:
        return t5_model, t5_tokenizer, DEVICE

    print("Loading T5 model and tokenizer.")
    t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    if t5_tokenizer.pad_token_id is None:
        t5_tokenizer.pad_token_id = t5_tokenizer.eos_token_id

    return t5_model, t5_tokenizer, DEVICE


def clear_memory():
    """
    Free up GPU cache and run Python garbage collection.
    """
    torch.cuda.empty_cache()
    gc.collect()


def summarize_text(text: str, idx: int = None, total: int = None) -> str:
    """
    Summarize a single piece of text using T5:
      1 Truncate raw text to first TRUNCATE_TEXT_CHARS chars
      2 Prepend "summarize: " so T5 uses its summarization head
      3 Generate with beam search, max length 150

    Returns the generated summary string.
    """
    model, tokenizer, device = load_t5_model()

    if idx is not None and total is not None:
        print(f"Summarizing [{idx}/{total}]")

    raw_snippet = text[:TRUNCATE_TEXT_CHARS]
    prompt = "summarize: " + raw_snippet

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SUMMARIZE_MAX_INPUT,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=SUMMARIZE_MAX_TARGET,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    clear_memory()
    return summary


def fine_tune_t5_on_papers(
    df,
    output_dir: str = r"C:\codes\t5-db\fine_tuned_t5"
) -> str:
    """
    Fine-tune T5 on a DataFrame with columns ['input_text', 'summary'].
    Saves the fine-tuned model + tokenizer to output_dir.
    """
    model, tokenizer, device = load_t5_model()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    required = {"input_text", "summary"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing column(s): {missing}")

    print("Fine-tuning T5 on provided data.")

    def tokenize_fn(batch):
        enc_in = tokenizer(
            batch["input_text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        enc_out = tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=150,
        )
        enc_in["labels"] = enc_out["input_ids"]
        return enc_in

    hf_ds = Dataset.from_pandas(df)
    tokenized = hf_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=list(df.columns),
        load_from_cache_file=False,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        save_steps=500,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        evaluation_strategy="no",
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        report_to=[],
        save_total_limit=1,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"T5 fine-tuned and saved to {output_dir}")
    clear_memory()
    return output_dir
