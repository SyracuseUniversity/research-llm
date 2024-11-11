# prepare_dataset.py

def prepare_dataset(pdf_texts):
    """Creates a dataset structure from PDF texts with placeholder summaries."""
    dataset = [{"text": text, "summary": "Placeholder summary"} for text in pdf_texts.values()]
    return dataset
