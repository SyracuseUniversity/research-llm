from PyPDF2 import PdfReader
import re

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return clean_text(text)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def clean_text(text):
    """Cleans extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove unwanted characters
    return text.strip()
