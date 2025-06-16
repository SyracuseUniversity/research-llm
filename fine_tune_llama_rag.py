"""
fine_tune_llama_rag.py  –  QLoRA‐based fine‐tuning of 4‐bit LLaMA (no emojis).
"""

import os
import gc
import torch
import random
import numpy as np
import pandas as pd

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)

# These will be overridden by run_pipeline.py:
QA_DATASET_PATH: str | None = None
OUTPUT_PATH:     str | None = None

BASE_MODEL_PATH = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
MAX_LEN = 256
SEED = 42


def _clean_mem():
    torch.cuda.empty_cache()
    gc.collect()


def load_llama_model():
    """
    Load the 4-bit LLaMA base + LoRA config.
    Returns (model, tokenizer, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global llama_model, llama_tokenizer
    try:
        llama_model
        llama_tokenizer
    except NameError:
        llama_model = None
        llama_tokenizer = None

    if llama_model is not None and llama_tokenizer is not None:
        return llama_model, llama_tokenizer, device

    print("🔧 Loading 4-bit LLaMA base model…")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    llama_model = get_peft_model(base_model, lora_cfg)
    llama_model.print_trainable_parameters()

    return llama_model, llama_tokenizer, device


def fine_tune_llama_on_papers(df: pd.DataFrame):
    """
    Fine-tune the LLaMA model (with LoRA) on a DataFrame of QA pairs:
      df['input_text'], df['target_text'].
    Saves to OUTPUT_PATH.
    """
    if QA_DATASET_PATH is None or OUTPUT_PATH is None:
        raise ValueError("QA_DATASET_PATH and OUTPUT_PATH must be set.")

    model, tokenizer, device = load_llama_model()

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    set_seed(SEED)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    def tok_batch(batch):
        inputs = batch["input_text"]
        targets = batch["target_text"]

        merged = [f"{inp} {tgt}" for inp, tgt in zip(inputs, targets)]
        enc = tokenizer(
            merged,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors=None,
        )

        prompt_enc = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors=None,
        )["input_ids"]

        input_ids_out = []
        attention_mask_out = []
        labels_out = []

        for merged_ids, merged_mask, prompt_ids in zip(
            enc["input_ids"], enc["attention_mask"], prompt_enc
        ):
            merged_ids = list(merged_ids)
            merged_mask = list(merged_mask)
            labels = merged_ids.copy()

            prompt_len = sum(1 for token_id in prompt_ids if token_id != tokenizer.pad_token_id)
            for i in range(prompt_len):
                labels[i] = -100

            input_ids_out.append(merged_ids)
            attention_mask_out.append(merged_mask)
            labels_out.append(labels)

        return {
            "input_ids": input_ids_out,
            "attention_mask": attention_mask_out,
            "labels": labels_out,
        }

    hf_ds = Dataset.from_pandas(df)
    ds_tok = hf_ds.map(
        tok_batch,
        batched=True,
        remove_columns=list(df.columns),
        load_from_cache_file=False,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        evaluation_strategy="no",
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"LoRA adapter and tokenizer saved to {OUTPUT_PATH}")

    _clean_mem()
