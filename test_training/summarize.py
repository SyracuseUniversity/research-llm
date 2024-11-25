# summarize.py

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_trained_model():
    """Loads the quantized LLaMA model and tokenizer."""
    model_path = r"C:\Users\arapte\Llama-3.1-8B-Instruct"  # Ensure this path is correct

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,       # Enable 8-bit quantization
        llm_int8_threshold=6.0,  # Default threshold
        llm_int8_has_fp16_weight=False
    )

    # Load the model with quantization and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )

    return tokenizer, model

def split_text_into_batches(text, tokenizer, max_input_length):
    """Splits text into batches that fit within the model's context window."""
    tokens = tokenizer.encode(text)
    batches = []
    for i in range(0, len(tokens), max_input_length):
        batch_tokens = tokens[i:i + max_input_length]
        batch_text = tokenizer.decode(batch_tokens)
        batches.append(batch_text)
    return batches

def summarize_text(text, tokenizer, model):
    """Generates a summary for long texts by processing batches."""
    device = next(model.parameters()).device  # Get device from model

    # Set pad_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_context_length = 2048

    max_new_tokens = 50  # Reduced to minimize memory usage

    # Calculate max allowed input length
    max_input_length = max_context_length - max_new_tokens - 50  # Extra buffer

    # Split the input text into batches
    batches = split_text_into_batches(text, tokenizer, max_input_length)

    summaries = []

    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")

        # Prepare the prompt
        prompt = f"Summarize the following text:\n{batch}\nSummary:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_length - max_new_tokens,
        ).to(device)

        # Generate the summary
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,  # Greedy decoding to reduce memory usage
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract the summary from the output
        summary = output_text[len(prompt):].strip()
        summaries.append(summary)

        # Free up memory
        del inputs
        del output_ids
        torch.cuda.empty_cache()

    # Combine summaries of all batches
    final_summary = ' '.join(summaries)
    return final_summary



