# main.py

import os
from extract_text import extract_texts_from_pdfs
from prepare_dataset import prepare_dataset
from tokenize_data import tokenize_dataset
from train_model import train_model
from test1.summarize import load_model_and_tokenizer, summarize_text

# Define paths
pdf_paths = [
    r"E:\OSPO\bioarxiv\2024.03.01.582919v1.full.pdf",
    r"E:\OSPO\eartharxiv\jones_et_al_2024.pdf",
    r"E:\OSPO\eartharxiv\jones_et_al_2022.pdf",
    r"E:\OSPO\eartharxiv\deeptime-preprint.pdf",
    r"E:\OSPO\evoarxiv\copy-of-pcm-sensitivity-manuscript.pdf"
]
model_path = r"D:\BACKUPLLAMA\Llama-3.1-8B-Instruct"

# Step 1: Extract text from PDFs
print("Extracting text from PDFs...")
pdf_texts = extract_texts_from_pdfs(pdf_paths)

# Step 2: Prepare dataset
print("Preparing dataset...")
dataset = prepare_dataset(pdf_texts)

# Step 3: Tokenize data
print("Tokenizing dataset...")
tokenized_dataset = tokenize_dataset(dataset, model_path)

# Step 4: Train model
print("Training model...")
train_model(tokenized_dataset, model_path)

# Step 5: Summarize new PDF
print("Summarizing a new PDF...")
model, tokenizer = load_model_and_tokenizer(model_path)
pdf_text = pdf_texts[pdf_paths[0]]  # Choose any PDF for testing
summary = summarize_text(pdf_text, model, tokenizer)
print("Summary:", summary)


