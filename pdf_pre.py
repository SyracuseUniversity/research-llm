"""
pdf_pre.py

This module provides functions for extracting and cleaning text from PDF files.
It uses PyPDF2 to read PDFs and includes a new function to extract researcher information
using basic heuristics.
It uses docling and markdowncleaner to extract and clean markdown from PDFs
"""

from PyPDF2 import PdfReader

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from markdowncleaner import MarkdownCleaner

import re

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    
    Iterates through each page, concatenates text, and cleans it.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def clean_text(text):
    """
    Cleans extracted text by removing extra whitespace and unwanted characters.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove unwanted characters
    return text.strip()

def extract_research_info_from_pdf(file_path):
    """
    Attempts to extract researcher information from a PDF file using basic heuristics.
    
    Assumptions:
      - The first line of the PDF is the work title.
      - Searches for lines starting with 'Author:' or 'Authors:' to extract authors.
      - Uses the first author as the primary researcher.
    
    Returns:
        A dictionary with keys:
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
    
    # Initialize researcher name and authors.
    researcher_name = ""
    authors = ""
    
    # Search for "Author:" or "Authors:" in the text.
    match_authors = re.search(r'Author[s]*:\s*(.+)', text, re.IGNORECASE)
    if match_authors:
        authors = match_authors.group(1).strip()
        # Use the first author as the primary researcher.
        researcher_name = authors.split(",")[0].strip()
    
    # Use the first 1000 characters as additional info.
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

def extract_markdown_from_pdf(file_path):
    """Extracts markdown from a PDF file."""
    # set backend to PyPdfium (avoids some artifacts, introduces some others; faster than default DoclingParseV2DocumentBackend)
    backend=PyPdfiumDocumentBackend 

    try:
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(backend=backend)
            }
        )
        result = doc_converter.convert(file_path)
        return result.document.export_to_markdown()
    
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def clean_markdown(text):
    """Cleans extracted markdown using markdowncleaner package with default configuration."""
    cleaner = MarkdownCleaner()
    return cleaner.clean_markdown_string(text)
