"""
pdf_pre.py  –  Extract text & simple metadata from PDF files (no emojis).
"""

import os
import re
import gc
from PyPDF2 import PdfReader


def extract_raw_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text (with newline characters) from a PDF.
    Returns a single string with page breaks preserved.
    Raises on failure.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{file_path}': {e}")

    raw_pages = []
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None
        raw_pages.append(page_text or "")

    return "\n\n".join(raw_pages)


def clean_text(text: str) -> str:
    """
    Collapse multiple whitespace into single spaces,
    remove any character that is not letter, digit, punctuation, or whitespace.
    """
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\/\-\(\)]", "", text)
    return text.strip()


def extract_research_info_from_pdf(file_path: str) -> dict | None:
    """
    Extract simple metadata from a PDF:
      • work_title: first nonempty line
      • authors: text after 'Author(s):' or fallback 'By <name>'
      • researcher_name: first comma‐separated author
      • info: first 1000 chars of cleaned full text

    Returns dict with keys {researcher_name, work_title, authors, info, full_text}, 
    or None if extraction fails.
    """
    try:
        raw_text = extract_raw_text_from_pdf(file_path)
    except Exception as e:
        print(f"Failed to read '{file_path}': {e}")
        return None

    if not raw_text.strip():
        return None

    raw_lines = [ln.strip() for ln in raw_text.splitlines()]
    # Title: first nonempty line
    work_title = ""
    for ln in raw_lines:
        if ln:
            work_title = ln
            break

    # Authors / researcher
    authors = ""
    researcher_name = ""
    snippet_for_author_search = "\n".join(raw_lines[:20])
    author_match = re.search(r"Authors?\s*[:\-\s]\s*(.+)", snippet_for_author_search, re.IGNORECASE)
    if not author_match:
        for ln in raw_lines[:3]:
            by_match = re.match(r"By\s+(.+)", ln, re.IGNORECASE)
            if by_match:
                authors = by_match.group(1).strip()
                break
    else:
        authors = author_match.group(1).strip()

    if authors:
        researcher_name = authors.split(",")[0].strip()

    cleaned_full = clean_text(raw_text)
    info = cleaned_full[:1000]

    work_title = clean_text(work_title)
    authors = clean_text(authors)
    researcher_name = clean_text(researcher_name)

    if any([work_title, authors, researcher_name]):
        return {
            "researcher_name": researcher_name,
            "work_title": work_title,
            "authors": authors,
            "info": info,
            "full_text": cleaned_full
        }
    return None
