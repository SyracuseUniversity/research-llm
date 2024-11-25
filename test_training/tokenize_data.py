# tokenize_data.py

from transformers import AutoTokenizer

def load_tokenizer(model_name):
    """Loads the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    """Tokenizes the dataset."""
    def preprocess_function(examples):
        inputs = ["Summarize: " + doc for doc in examples["text"]]
        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding='max_length'
        )
        labels = tokenizer(
            examples["summary"], max_length=150, truncation=True, padding='max_length'
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_data = dataset.map(preprocess_function, batched=True)
    return tokenized_data
