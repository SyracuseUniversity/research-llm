# tokenize_data.py

from transformers import LLaMATokenizer

def load_tokenizer(model_path):
    """Loads the LLaMA tokenizer."""
    tokenizer = LLaMATokenizer.from_pretrained(model_path)
    return tokenizer

def preprocess_function(examples, tokenizer):
    """Preprocesses text for training."""
    inputs = tokenizer("summarize: " + examples["text"], max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=256, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

def tokenize_dataset(dataset, model_path):
    """Tokenizes the dataset."""
    tokenizer = load_tokenizer(model_path)
    tokenized_data = [preprocess_function(item, tokenizer) for item in dataset]
    return tokenized_data
