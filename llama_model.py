# # # llama_model.py
# # import torch
# # import gc
# # from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
# # from datasets import Dataset
# # import os

# # # Define base and fine-tuned model paths.
# # BASE_MODEL_PATH = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
# # FINE_TUNED_MODEL_PATH = r"C:\codes\llama32\fine_tuned_llama"

# # print("Using base model from:", BASE_MODEL_PATH)
# # print("Fine-tuned model will be saved to:", FINE_TUNED_MODEL_PATH)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Load the model: If fine-tuned directory exists, load from there; otherwise load base model.
# # if os.path.isdir(FINE_TUNED_MODEL_PATH):
# #     print("Loading fine-tuned LLaMA model from:", FINE_TUNED_MODEL_PATH)
# #     llama_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)
# # else:
# #     print("Fine-tuned model directory not found; loading base model from:", BASE_MODEL_PATH)
# #     llama_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)

# # llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# # llama_tokenizer.pad_token = llama_tokenizer.eos_token
# # print("Manually setting pad token to eos token.")

# # if hasattr(llama_model, "enable_gradient_checkpointing"):
# #     llama_model.enable_gradient_checkpointing()
# #     llama_model.config.use_cache = False

# # def clear_memory():
# #     """Clear GPU and CPU memory."""
# #     torch.cuda.empty_cache()
# #     gc.collect()

# # def chatbot_answer(question):
# #     """
# #     Generates an answer for the given research question using the fine-tuned LLaMA model.
    
# #     The prompt instructs the model to answer concisely and accurately.
# #     After generation, the output is post-processed to return only the final answer,
# #     ensuring that prompt instructions are removed.
# #     """
# #     prompt = (
# #         "You are a knowledgeable research assistant trained on a large corpus of research papers. "
# #         "Answer the following research question concisely and accurately, using only your learned knowledge. "
# #         "Ensure your answer is complete and ends with a proper punctuation mark (e.g., '.', '?', or '!').\n\n"
# #         "Question: " + question + "\n\n"
# #         "Answer:"
# #     )
# #     inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
# #     outputs = llama_model.generate(
# #         **inputs,
# #         max_new_tokens=300,
# #         do_sample=True,
# #         temperature=0.7,
# #         top_p=0.9,
# #         num_return_sequences=1,
# #         eos_token_id=llama_tokenizer.eos_token_id,
# #     )
# #     clear_memory()
# #     generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
# #     # Post-process: split on "Answer:" and take the part after the last occurrence.
# #     if "Answer:" in generated_text:
# #         final_answer = generated_text.split("Answer:")[-1].strip()
# #     else:
# #         final_answer = generated_text.strip()
    
# #     # Ensure the final answer ends with proper punctuation.
# #     if not final_answer.endswith(('.', '?', '!')):
# #         final_answer += "."
    
# #     return final_answer

# # def fine_tune_llama_on_papers(dataset, output_dir=FINE_TUNED_MODEL_PATH):
# #     """
# #     Fine-tunes the LLaMA model using paired data:
# #       - Input: full paper text.
# #       - Target: T5-generated summary.
      
# #     Constructs a combined sequence in the format:
# #       "Paper: {full_text}\nSummary: {summary}"
# #     with prompt tokens masked so that only the summary contributes to the loss.
# #     """
# #     if 'input_text' not in dataset.columns or 'target_text' not in dataset.columns:
# #         raise ValueError("Dataset must contain 'input_text' and 'target_text' columns!")
    
# #     def tokenize_function(examples):
# #         combined_texts = []
# #         labels_list = []
# #         for full_text, summary in zip(examples['input_text'], examples['target_text']):
# #             prompt = "Paper: " + full_text + "\nSummary:"
# #             combined = prompt + " " + summary
# #             tokenized = llama_tokenizer(combined, truncation=True, padding="max_length", max_length=256)
# #             combined_ids = tokenized['input_ids']
# #             prompt_ids = llama_tokenizer(prompt, truncation=True, padding=False)['input_ids']
# #             prompt_length = len(prompt_ids)
# #             label_ids = combined_ids.copy()
# #             for i in range(min(prompt_length, len(label_ids))):
# #                 label_ids[i] = -100
# #             combined_texts.append(combined_ids)
# #             labels_list.append(label_ids)
# #         attention_masks = [
# #             [1 if token_id != llama_tokenizer.pad_token_id else 0 for token_id in ids]
# #             for ids in combined_texts
# #         ]
# #         return {"input_ids": combined_texts, "attention_mask": attention_masks, "labels": labels_list}
    
# #     hf_dataset = Dataset.from_pandas(dataset)
# #     tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, batch_size=8)
    
# #     training_args = TrainingArguments(
# #         output_dir=output_dir,
# #         num_train_epochs=3,
# #         per_device_train_batch_size=8,
# #         gradient_accumulation_steps=1,
# #         learning_rate=3e-5,
# #         lr_scheduler_type="cosine",
# #         warmup_steps=500,
# #         weight_decay=0.01,
# #         save_steps=1000,
# #         save_total_limit=2,
# #         logging_steps=100,
# #         fp16=True,
# #         evaluation_strategy="steps",
# #         eval_steps=100,
# #         load_best_model_at_end=True,
# #         dataloader_num_workers=4
# #     )

# #     trainer = Trainer(
# #         model=llama_model,
# #         args=training_args,
# #         train_dataset=tokenized_dataset,
# #         eval_dataset=tokenized_dataset,
# #         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
# #     )

# #     os.makedirs(output_dir, exist_ok=True)
    
# #     checkpoint = None
# #     if os.path.isdir(output_dir):
# #         checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
# #         if checkpoints:
# #             checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
# #             print(f"Resuming training from checkpoint: {checkpoint}")

# #     print("Starting fine-tuning for LLaMA model...")
# #     trainer.train(resume_from_checkpoint=checkpoint)
# #     print(f"Saving fine-tuned LLaMA model to {output_dir}...")
# #     try:
# #         llama_model.save_pretrained(output_dir, use_safetensors=False)
# #     except Exception as e:
# #         print("Error during save_pretrained:", e)
# #         print("Falling back to manual saving of config and state_dict.")
# #         llama_model.config.save_pretrained(output_dir)
# #         torch.save(llama_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
# #     clear_memory()
# #     return output_dir


# # train_llama.py
# import sqlite3
# import pandas as pd
# import pickle

# def load_training_data_from_db(db_path):
#     """
#     Loads training data from the specified database by combining data from two tables:
    
#     1. The 'works' table: contains full_text and summary (T5-generated).
#        - Input: full_text
#        - Target: summary
       
#     2. The 'research_info' table: contains researcher_name, affiliation, work_title, authors, and info.
#        - Input: A formatted string with researcher details.
#        - Target: info (the correct researcher/work information)
       
#     Returns a DataFrame with columns 'input_text' and 'target_text'.
#     """
#     conn = sqlite3.connect(db_path)
    
#     # Load data from the 'works' table.
#     query_works = """
#         SELECT full_text, summary 
#         FROM works
#         WHERE summary_status = 'summarized' AND progress = 1
#     """
#     df_works = pd.read_sql_query(query_works, conn)
#     df_works = df_works.rename(columns={"full_text": "input_text", "summary": "target_text"})
    
#     # Load data from the 'research_info' table.
#     query_info = """
#         SELECT researcher_name, affiliation, work_title, authors, info
#         FROM research_info
#     """
#     df_info = pd.read_sql_query(query_info, conn)
#     conn.close()
    
#     if not df_info.empty:
#         # Construct training pairs for research info.
#         df_info["input_text"] = (
#             "Researcher: " + df_info["researcher_name"].fillna("") + "\n" +
#             "Affiliation: " + df_info["affiliation"].fillna("") + "\n" +
#             "Work Title: " + df_info["work_title"].fillna("") + "\n" +
#             "Authors: " + df_info["authors"].fillna("") + "\n"
#         )
#         df_info["target_text"] = df_info["info"]
#         df_info = df_info[["input_text", "target_text"]]
#         # Combine both datasets.
#         df_combined = pd.concat([df_works, df_info], ignore_index=True)
#     else:
#         df_combined = df_works
    
#     return df_combined

# def process_and_save_data(processed_file, db_path):
#     """
#     Loads training data from the database (combining 'works' and 'research_info') and saves it as a pickle file.
#     """
#     print("Loading training data from database...")
#     df = load_training_data_from_db(db_path)
#     with open(processed_file, "wb") as f:
#         pickle.dump(df, f)
#     print("Processed training data saved to file:", processed_file)
#     return df

# if __name__ == "__main__":
#     processed_file = r"C:\codes\t5-db\processed_training_data.pkl"
#     db_path = r"C:\codes\t5-db\researchers.db"
    
#     df = process_and_save_data(processed_file, db_path)
#     print(f"Retrieved {len(df)} training examples from the database.")
    
#     from llama_model import fine_tune_llama_on_papers
#     output_dir = fine_tune_llama_on_papers(df)
#     print("Fine-tuned LLaMA model saved at:", output_dir)

# llama_model.py
# import torch
# import gc
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
# from datasets import Dataset
# import os

# # Define base and fine-tuned model paths.
# BASE_MODEL_PATH = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
# FINE_TUNED_MODEL_PATH = r"C:\codes\llama32\fine_tuned_llama"

# print("Using base model from:", BASE_MODEL_PATH)
# print("Fine-tuned model will be saved to:", FINE_TUNED_MODEL_PATH)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # If the fine-tuned model directory exists, load from it; otherwise, load the base model.
# if os.path.isdir(FINE_TUNED_MODEL_PATH):
#     print("Loading fine-tuned LLaMA model from:", FINE_TUNED_MODEL_PATH)
#     llama_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)
# else:
#     print("Fine-tuned model directory not found; loading base model from:", BASE_MODEL_PATH)
#     llama_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)

# llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# llama_tokenizer.pad_token = llama_tokenizer.eos_token
# print("Manually setting pad token to eos token.")

# if hasattr(llama_model, "enable_gradient_checkpointing"):
#     llama_model.enable_gradient_checkpointing()
#     llama_model.config.use_cache = False

# def clear_memory():
#     """Clear GPU and CPU memory."""
#     torch.cuda.empty_cache()
#     gc.collect()

# def chatbot_answer(question):
#     """
#     Generates an answer for the given research question using the fine-tuned LLaMA model.
#     The prompt instructs the model to answer using only its learned knowledge.
#     After generation, the function post-processes the output to return only the final answer.
#     """
#     prompt = (
#         "You are a knowledgeable research assistant trained on a large corpus of research papers. "
#         "Answer the following research question concisely and accurately, using only your learned knowledge. "
#         "Ensure your answer is complete and ends with proper punctuation (e.g., '.', '?', or '!').\n\n"
#         "Question: " + question + "\n\n"
#         "Answer:"
#     )
#     inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
#     outputs = llama_model.generate(
#         **inputs,
#         max_new_tokens=300,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         num_return_sequences=1,
#         eos_token_id=llama_tokenizer.eos_token_id,
#     )
#     clear_memory()
#     generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Post-process: remove the prompt by splitting at "Answer:" and taking the text after.
#     if "Answer:" in generated_text:
#         final_answer = generated_text.split("Answer:")[-1].strip()
#     else:
#         final_answer = generated_text.strip()
#     if not final_answer.endswith((".", "?", "!")):
#         final_answer += "."
#     return final_answer

# # def fine_tune_llama_on_papers(dataset, output_dir=FINE_TUNED_MODEL_PATH):
# #     """
# #     Fine-tunes the LLaMA model using paired data from two sources:
# #       - Input: full paper text (from the 'works' table) or a formatted researcher info string (from the 'research_info' table).
# #       - Target: the corresponding summary (T5-generated or the correct info).
      
# #     The training sequence is constructed as:
# #       "Paper: {input_text}\nSummary: {target_text}"
# #     with the prompt portion masked so that loss is computed only on the summary tokens.
# #     """
# #     if 'input_text' not in dataset.columns or 'target_text' not in dataset.columns:
# #         raise ValueError("Dataset must contain 'input_text' and 'target_text' columns!")
    
# #     def tokenize_function(examples):
# #         combined_texts = []
# #         labels_list = []
# #         for inp, tgt in zip(examples['input_text'], examples['target_text']):
# #             prompt = "Paper: " + inp + "\nSummary:"
# #             combined = prompt + " " + tgt
# #             tokenized = llama_tokenizer(combined, truncation=True, padding="max_length", max_length=256)
# #             combined_ids = tokenized['input_ids']
# #             prompt_ids = llama_tokenizer(prompt, truncation=True, padding=False)['input_ids']
# #             prompt_length = len(prompt_ids)
# #             label_ids = combined_ids.copy()
# #             for i in range(min(prompt_length, len(label_ids))):
# #                 label_ids[i] = -100
# #             combined_texts.append(combined_ids)
# #             labels_list.append(label_ids)
# #         attention_masks = [
# #             [1 if token_id != llama_tokenizer.pad_token_id else 0 for token_id in ids]
# #             for ids in combined_texts
# #         ]
# #         return {"input_ids": combined_texts, "attention_mask": attention_masks, "labels": labels_list}
    
# #     hf_dataset = Dataset.from_pandas(dataset)
# #     tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, batch_size=8)
    
# #     training_args = TrainingArguments(
# #         output_dir=output_dir,
# #         num_train_epochs=3,
# #         per_device_train_batch_size=8,
# #         gradient_accumulation_steps=1,
# #         learning_rate=3e-5,
# #         lr_scheduler_type="cosine",
# #         warmup_steps=500,
# #         weight_decay=0.01,
# #         save_steps=1000,
# #         save_total_limit=2,
# #         logging_steps=100,
# #         fp16=True,
# #         evaluation_strategy="steps",
# #         eval_steps=100,
# #         load_best_model_at_end=True,
# #         dataloader_num_workers=4
# #     )

# #     trainer = Trainer(
# #         model=llama_model,
# #         args=training_args,
# #         train_dataset=tokenized_dataset,
# #         eval_dataset=tokenized_dataset,
# #         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
# #     )

# #     os.makedirs(output_dir, exist_ok=True)
    
# #     checkpoint = None
# #     if os.path.isdir(output_dir):
# #         checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
# #         if checkpoints:
# #             checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
# #             print(f"Resuming training from checkpoint: {checkpoint}")

# #     print("Starting fine-tuning for LLaMA model...")
# #     trainer.train(resume_from_checkpoint=checkpoint)
# #     print(f"Saving fine-tuned LLaMA model to {output_dir}...")
# #     try:
# #         llama_model.save_pretrained(output_dir, use_safetensors=False)
# #     except Exception as e:
# #         print("Error during save_pretrained:", e)
# #         print("Falling back to manual saving of config and state_dict.")
# #         llama_model.config.save_pretrained(output_dir)
# #         torch.save(llama_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
# #     clear_memory()
# #     return output_dir

# llama_model.py

import os
import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset

# Constants
BASE_MODEL_PATH = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
FINE_TUNED_MODEL_PATH = r"C:\codes\llama32\fine_tuned_llama"

# Globals (lazy loading)
llama_model = None
llama_tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_llama_model():
    """Loads the LLaMA model and tokenizer only once."""
    global llama_model, llama_tokenizer

    if llama_model is not None and llama_tokenizer is not None:
        return llama_model, llama_tokenizer, device

    print("Using base model from:", BASE_MODEL_PATH)
    print("Fine-tuned model will be saved to:", FINE_TUNED_MODEL_PATH)

    model_file = os.path.join(FINE_TUNED_MODEL_PATH, "pytorch_model.bin")
    if os.path.exists(model_file):
        print("✅ Fine-tuned model found. Loading from:", FINE_TUNED_MODEL_PATH)
        llama_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)
    else:
        print("⚠️ Fine-tuned model not found. Loading base model from:", BASE_MODEL_PATH)
        llama_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)

    llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print("Manually setting pad token to eos token.")

    if hasattr(llama_model, "enable_gradient_checkpointing"):
        llama_model.enable_gradient_checkpointing()
        llama_model.config.use_cache = False

    return llama_model, llama_tokenizer, device

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def chatbot_answer(question):
    """Generate an answer for a Syracuse-specific research question."""
    model, tokenizer, device = load_llama_model()

    prompt = (
        "You are a Syracuse University research assistant trained to answer questions about Syracuse-affiliated authors, papers, and subjects.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    clear_memory()
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    else:
        return generated_text.strip()

def fine_tune_llama_on_papers(dataset, output_dir=FINE_TUNED_MODEL_PATH):
    """
    Fine-tune LLaMA using Syracuse metadata (input_text, target_text pairs).
    """
    model, tokenizer, device = load_llama_model()

    if 'input_text' not in dataset.columns or 'target_text' not in dataset.columns:
        raise ValueError("Dataset must contain 'input_text' and 'target_text' columns.")

    def tokenize_function(examples):
        combined_inputs = []
        labels_list = []
        for prompt, target in zip(examples["input_text"], examples["target_text"]):
            combined = prompt + " " + target
            tokenized = tokenizer(combined, truncation=True, padding="max_length", max_length=256)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Mask the prompt in the labels
            prompt_ids = tokenizer(prompt, truncation=True, padding=False)["input_ids"]
            label_ids = input_ids.copy()
            for i in range(min(len(prompt_ids), len(label_ids))):
                label_ids[i] = -100

            combined_inputs.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids
            })

        batch = {
            "input_ids": [x["input_ids"] for x in combined_inputs],
            "attention_mask": [x["attention_mask"] for x in combined_inputs],
            "labels": [x["labels"] for x in combined_inputs],
        }
        return batch

    hf_dataset = Dataset.from_pandas(dataset)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, batch_size=8)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        eval_steps=100,
        evaluation_strategy="steps",
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        learning_rate=2e-5,
        warmup_steps=300,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🧠 Starting fine-tuning of LLaMA on metadata...")

    # Check for resume checkpoint
    checkpoint_dir = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint_dir = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"🔄 Resuming from checkpoint: {checkpoint_dir}")

    trainer.train(resume_from_checkpoint=checkpoint_dir)

    print(f"💾 Saving fine-tuned LLaMA model to {output_dir}...")
    try:
        model.save_pretrained(output_dir, use_safetensors=False)
    except Exception as e:
        print("Error using save_pretrained:", e)
        print("Saving config and weights manually.")
        model.config.save_pretrained(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    clear_memory()
    return output_dir

