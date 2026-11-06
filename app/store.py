"""Embeddings + ChromaDB vector store."""
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models work best when queries (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_client = chromadb.PersistentClient(path="data/chroma")
collection = _client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # Loaded once, on first use (the first run downloads the model)
    return SentenceTransformer(EMBED_MODEL)


def add_document(doc_id: str, filename: str, chunks: list[dict]) -> None:
    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts, normalize_embeddings=True).tolist()
    collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"doc_id": doc_id, "filename": filename, "page": c["page"]} for c in chunks
        ],
    )


def search(question: str, k: int = 5, doc_id: str | None = None) -> list[dict]:
    q_emb = get_model().encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    kwargs = {"query_embeddings": q_emb, "n_results": k}
    if doc_id:
        kwargs["where"] = {"doc_id": doc_id}
    res = collection.query(**kwargs)
    return [
        {"text": t, "page": m["page"], "filename": m["filename"], "distance": d}
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def list_documents() -> list[dict]:
    metas = collection.get(include=["metadatas"])["metadatas"]
    seen = {m["doc_id"]: m["filename"] for m in metas}
    return [{"doc_id": d, "filename": f} for d, f in sorted(seen.items(), key=lambda x: x[1])]


def delete_document(doc_id: str) -> None:
    collection.delete(where={"doc_id": doc_id})
