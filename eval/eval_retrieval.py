"""Measure retrieval hit rate for different chunk sizes and k.

Usage:
    python eval/eval_retrieval.py path/to/test.pdf eval/questions.json

questions.json format:
    [{"question": "What is the refund period?", "page": 4}, ...]

Hit = the correct page appears among the top-k retrieved chunks.
Self-contained: uses numpy cosine similarity, no ChromaDB needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ingest import chunk_pages, extract_pages  # noqa: E402
from app.store import EMBED_MODEL, QUERY_PREFIX  # noqa: E402

CHUNK_SIZES = [500, 800, 1200]
KS = [3, 5, 8]


def main(pdf_path: str, questions_path: str) -> None:
    questions = json.loads(Path(questions_path).read_text())
    pages = extract_pages(pdf_path)
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([QUERY_PREFIX + q["question"] for q in questions], normalize_embeddings=True)

    print(f"{'chunk_size':>10} | " + " | ".join(f"k={k}" for k in KS))
    print("-" * 40)
    for size in CHUNK_SIZES:
        chunks = chunk_pages(pages, chunk_size=size, overlap=size // 5)
        c_emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
        sims = q_emb @ c_emb.T  # cosine similarity (vectors are normalized)
        row = []
        for k in KS:
            hits = 0
            for qi, q in enumerate(questions):
                top = np.argsort(-sims[qi])[:k]
                if any(chunks[j]["page"] == q["page"] for j in top):
                    hits += 1
            row.append(f"{hits / len(questions):.0%}".rjust(4))
        print(f"{size:>10} | " + " | ".join(row))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
