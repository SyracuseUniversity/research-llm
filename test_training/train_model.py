# train_model.py

import os
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch
from tokenize_data import load_tokenizer, tokenize_dataset
from preprocessing import extract_text_from_pdf, preprocess_text

def prepare_dataset(pdf_paths):
    """Prepares the dataset from PDFs."""
    texts = []
    summaries = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        if text:
            cleaned_text = preprocess_text(text)
            summary_sentences = cleaned_text.split('. ')[:5]
            summary = '. '.join(summary_sentences)
            texts.append(cleaned_text)
            summaries.append(summary)
        else:
            print(f"No text found in {pdf_path}.")
    dataset = Dataset.from_dict({"text": texts, "summary": summaries})
    return dataset

def load_model(model_name):
    """Loads the model."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def train_model():
    """Prepares data, trains the model, and saves it."""
    pdf_paths = [
        r"C:\codes\test_training\train_pdfs\jones_et_al_2022.pdf",
        r"C:\codes\test_training\train_pdfs\jones_et_al_2024.pdf",
        r"C:\codes\test_training\train_pdfs\deeptime-preprint.pdf",
        r"C:\codes\test_training\train_pdfs\copy-of-pcm-sensitivity-manuscript.pdf",
        r"C:\codes\test_training\train_pdfs\2024.03.01.582919v1.full.pdf"
    ]

    # Prepare the dataset
    dataset = prepare_dataset(pdf_paths)

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.2)

    # Load tokenizer and model
    model_name = 't5-base'  
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./trained_model",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    output_dir = './trained_model'
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model training complete and saved.")

if __name__ == '__main__':
    train_model()
