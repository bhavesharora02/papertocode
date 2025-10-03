import pytesseract
from pdf2image import convert_from_path
from typing import List

def ocr_pdf(pdf_path: str) -> List[str]:
    """Convert PDF pages to images, run OCR"""
    pages = convert_from_path(pdf_path, dpi=200)
    texts = [pytesseract.image_to_string(img) for img in pages]
    return texts
