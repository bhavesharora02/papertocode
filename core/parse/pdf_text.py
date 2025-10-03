import pdfplumber
from typing import Dict

SECTIONS = ["abstract", "introduction", "related work", "method", "model", "algorithm", "experiments", "results"]

def extract_text(pdf_path: str) -> str:
    """Extract raw text from PDF using pdfplumber"""
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def chunk_sections(raw_text: str) -> Dict[str, str]:
    """Naive split of text into sections"""
    chunks = {}
    lower_text = raw_text.lower()
    for sec in SECTIONS:
        if sec in lower_text:
            idx = lower_text.index(sec)
            chunks[sec] = raw_text[idx: idx + 2000]  # crude slice (improve later)
    return chunks
