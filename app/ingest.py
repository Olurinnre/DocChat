"""PDF -> pages -> chunks (every chunk remembers its page number)."""
from pypdf import PdfReader


def extract_pages(path: str) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...]. Pages with no text are skipped."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Fixed-size character chunks with overlap, chunked per page so citations stay exact."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"]})
            start += chunk_size - overlap
    return chunks
