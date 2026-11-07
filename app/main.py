"""FastAPI backend: /upload, /ask, /documents."""
import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .ingest import chunk_pages, extract_pages
from .rag import answer
from .store import add_document, delete_document, list_documents

MAX_UPLOAD_MB = 25
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Chat With Your Documents")


class Question(BaseModel):
    question: str
    doc_id: str | None = None
    k: int = 5


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = uuid.uuid4().hex[:8]
    path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if os.path.getsize(path) > MAX_UPLOAD_MB * 1024 * 1024:
        os.remove(path)
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB).")

    pages = extract_pages(path)
    if not pages:
        os.remove(path)
        raise HTTPException(422, "No extractable text found (scanned PDF? OCR is not supported yet).")

    chunks = chunk_pages(pages)
    add_document(doc_id, file.filename, chunks)
    return {"doc_id": doc_id, "filename": file.filename, "pages": len(pages), "chunks": len(chunks)}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(400, "Question is empty.")
    try:
        return answer(q.question, doc_id=q.doc_id, k=q.k)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")

@app.get("/documents")
def documents():
    return list_documents()


@app.delete("/documents/{doc_id}")
def remove(doc_id: str):
    delete_document(doc_id)
    path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(path):
        os.remove(path)
    return {"deleted": doc_id}
