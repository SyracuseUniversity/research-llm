# # pdf_pre.py
# from PyPDF2 import PdfReader
# import re

# def extract_text_from_pdf(file_path):
#     """Extracts text from a PDF file."""
#     text = ""
#     try:
#         reader = PdfReader(file_path)
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#         return clean_text(text)
#     except Exception as e:
#         print(f"Failed to process {file_path}: {e}")
#         return None

# def clean_text(text):
#     """Cleans extracted text."""
#     text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
#     text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove unwanted characters
#     return text.strip()

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

def extract_research_info_from_pdf(file_path):
    """
    Attempts to extract researcher information from a PDF file using basic heuristics.
    Assumes:
      - The first line is the work title.
      - Looks for lines starting with 'Author:' or 'Authors:' to extract authors.
    
    Returns a dictionary with keys:
      researcher_name, work_title, authors, info
    or None if no relevant information is found.
    """
    text = extract_text_from_pdf(file_path)
    if not text:
        return None

    # Split text into lines.
    lines = text.split("\n")
    
    # Assume the first line is the work title.
    work_title = lines[0].strip() if lines else ""
    
    # Initialize fields.
    researcher_name = ""
    authors = ""
    
    # Search for "Author:" or "Authors:".
    match_authors = re.search(r'Author[s]*:\s*(.+)', text, re.IGNORECASE)
    if match_authors:
        authors = match_authors.group(1).strip()
        # Assume the first author is the primary researcher.
        researcher_name = authors.split(",")[0].strip()
    
    # Use the first 1000 characters as fallback for additional info.
    info = text[:1000]
    
    if work_title or researcher_name or authors:
        return {
            "researcher_name": researcher_name,
            "work_title": work_title,
            "authors": authors,
            "info": info
        }
    else:
        return None
