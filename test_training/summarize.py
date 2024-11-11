# summarize.py

from transformers import LLaMATokenizer, LLaMAModel

def load_model_and_tokenizer(model_path):
    """Loads the LLaMA model and tokenizer."""
    model = LLaMAModel.from_pretrained(model_path)
    tokenizer = LLaMATokenizer.from_pretrained(model_path)
    return model, tokenizer

def summarize_text(text, model, tokenizer):
    """Generates a summary for the input text."""
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
