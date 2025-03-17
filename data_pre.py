"""
data_pre.py

This module prepares text for T5 summarization.
It adds the "summarize:" prefix, truncates the input to 1000 characters,
and tokenizes the text using the T5Tokenizer.
"""

from transformers import T5Tokenizer

def preprocess_text_for_t5(text, model_name="t5-small"):
    """Prepares text for T5 summarization."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # Preprocess text with task prefix and truncate to the first 1000 characters.
    input_text = "summarize: " + text[:1000]
    return tokenizer(input_text, truncation=True, max_length=512, padding="max_length")
