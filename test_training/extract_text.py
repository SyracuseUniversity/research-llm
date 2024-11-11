# extract_text.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_texts_from_pdfs(pdf_paths):
    """Extracts text from a list of PDF paths."""
    pdf_texts = {pdf_path: extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths}
    return pdf_texts
