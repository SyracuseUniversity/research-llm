# preprocessing.py

import PyPDF2
import nltk
import string
import re

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def preprocess_text(text):
    """Cleans and preprocesses text."""
    # Split text into sentences using regular expressions
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Remove punctuation and make lowercase
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        cleaned_sentences.append(sentence)

    # Reconstruct the text
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text
