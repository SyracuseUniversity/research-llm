import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import os

# Build relative paths to the LLaMA base model and fine-tuned model directories
script_dir = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL_PATH = os.path.join(script_dir, "Llama-3.2-1B-Instruct")  # Put your base model folder here
FINE_TUNED_MODEL_PATH = os.path.join(script_dir, "fine_tuned_llama")

print("Using base model from:", BASE_MODEL_PATH)
print("Fine-tuned model will be saved to:", FINE_TUNED_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model: If fine-tuned directory exists, load from there; otherwise load base model.
if os.path.isdir(FINE_TUNED_MODEL_PATH):
    print("Loading fine-tuned LLaMA model from:", FINE_TUNED_MODEL_PATH)
    llama_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)
else:
    print("Fine-tuned model directory not found; loading base model from:", BASE_MODEL_PATH)
    llama_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)

llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
print("Manually setting pad token to eos token.")

if hasattr(llama_model, "enable_gradient_checkpointing"):
    llama_model.enable_gradient_checkpointing()
    llama_model.config.use_cache = False

def clear_memory():
    """Clear GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()

def chatbot_answer(question):
    """
    Generates an answer for the given research question using the fine-tuned LLaMA model.
    
    The prompt instructs the model to answer concisely and accurately.
    """
    prompt = (
        "You are a knowledgeable research assistant trained on a large corpus of research papers. "
        "Answer the following research question concisely and accurately, using only your learned knowledge. "
        "Ensure your answer is complete and ends with a proper punctuation mark (e.g., '.', '?', or '!').\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = llama_model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=llama_tokenizer.eos_token_id,
    )
    clear_memory()
    generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process: split on "Answer:" and take the part after the last occurrence.
    if "Answer:" in generated_text:
        final_answer = generated_text.split("Answer:")[-1].strip()
    else:
        final_answer = generated_text.strip()
    
    # Ensure the final answer ends with proper punctuation.
    if not final_answer.endswith(('.', '?', '!')):
        final_answer += "."
    
    return final_answer

def fine_tune_llama_on_papers(dataset, output_dir=FINE_TUNED_MODEL_PATH):
    """
    Fine-tunes the LLaMA model using paired data:
      - Input: full paper text.
      - Target: T5-generated summary.
    """
    if 'input_text' not in dataset.columns or 'target_text' not in dataset.columns:
        raise ValueError("Dataset must contain 'input_text' and 'target_text' columns!")
    
    def tokenize_function(examples):
        combined_texts = []
        labels_list = []
        for full_text, summary in zip(examples['input_text'], examples['target_text']):
            prompt = "Paper: " + full_text + "\nSummary:"
            combined = prompt + " " + summary
            tokenized = llama_tokenizer(combined, truncation=True, padding="max_length", max_length=256)
            combined_ids = tokenized['input_ids']
            prompt_ids = llama_tokenizer(prompt, truncation=True, padding=False)['input_ids']
            prompt_length = len(prompt_ids)
            label_ids = combined_ids.copy()
            for i in range(min(prompt_length, len(label_ids))):
                label_ids[i] = -100
            combined_texts.append(combined_ids)
            labels_list.append(label_ids)
        attention_masks = [
            [1 if token_id != llama_tokenizer.pad_token_id else 0 for token_id in ids]
            for ids in combined_texts
        ]
        return {"input_ids": combined_texts, "attention_mask": attention_masks, "labels": labels_list}
    
    hf_dataset = Dataset.from_pandas(dataset)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, batch_size=8)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=llama_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"Resuming training from checkpoint: {checkpoint}")

    print("Starting fine-tuning for LLaMA model...")
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"Saving fine-tuned LLaMA model to {output_dir}...")
    try:
        llama_model.save_pretrained(output_dir, use_safetensors=False)
    except Exception as e:
        print("Error during save_pretrained:", e)
        print("Falling back to manual saving of config and state_dict.")
        llama_model.config.save_pretrained(output_dir)
        torch.save(llama_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    clear_memory()
    return output_dir
