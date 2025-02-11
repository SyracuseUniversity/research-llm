from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import gc
import os

model_name = "t5-small"
print("Loading T5 model on GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

def clear_memory():
    """Clear GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared memory and cache.")

def summarize_text(text, idx=None, total=None):
    """Generates a summary for the given text using T5."""
    if idx is not None and total is not None:
        print(f"Summarizing text [{idx}/{total}]...")
    input_text = "summarize: " + text[:1000]  # Truncate to the first 1000 characters
    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    clear_memory()  # Clear memory after summarization
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def fine_tune_t5_on_papers(dataset, output_dir=None):
    """Fine-tune the T5 model on the papers dataset."""
    if output_dir is None:
        # Build a relative path for the fine-tuned T5 model folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "fine_tuned_t5")

    print("Preparing dataset for fine-tuning...")

    # Ensure columns exist
    if 'input_text' not in dataset.columns or 'summary' not in dataset.columns:
        raise ValueError("Dataset must contain 'input_text' and 'summary' columns!")

    def tokenize_function(examples):
        inputs = t5_tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
        targets = t5_tokenizer(examples['summary'], padding="max_length", truncation=True, max_length=150)
        inputs['labels'] = targets['input_ids']
        return inputs

    hf_dataset = Dataset.from_pandas(dataset)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=2,         # Limit the number of checkpoints
        save_steps=500,             # Save model every 500 steps
        evaluation_strategy="no",   # Disable evaluation
        logging_dir="./logs",
        logging_steps=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,                  # Enable mixed precision
    )

    trainer = Trainer(
        model=t5_model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    print("Starting fine-tuning on GPU...")
    trainer.train()

    print(f"Saving the fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Model fine-tuned and saved successfully!")
    clear_memory()

    return output_dir
