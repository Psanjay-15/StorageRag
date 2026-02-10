import pymupdf
from io import BytesIO
from typing import Dict, Any


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Parse PDF from memory (bytes) → extract text only
    Returns plain text with page markers
    """
    all_text = []

    try:
        # Open PDF directly from bytes (no file on disk)
        stream = BytesIO(pdf_bytes)
        with pymupdf.open(stream=stream, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{text}")

        return "\n\n".join(all_text)

    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF from bytes: {str(e)}")
