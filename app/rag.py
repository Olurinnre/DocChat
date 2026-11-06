"""Retrieval + prompt design + Gemini call."""
from dotenv import load_dotenv
#import anthropic
import os
from openai import OpenAI


from .store import search

load_dotenv()
#llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
#MODEL = "claude-sonnet-5-5"


llm = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
MODEL = os.getenv("LLM_MODEL")

SYSTEM = """You answer questions using ONLY the numbered sources provided.
- Cite sources inline like [1], [2] right after each claim they support.
- If the sources do not contain the answer, reply exactly: "I couldn't find that in the document."
- Never use outside knowledge, and never invent citations.
- Be concise."""


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (File: {h['filename']}, Page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def answer(question: str, doc_id: str | None = None, k: int = 5, history: list[dict] | None = None) -> dict:
    hits = search(question, k=k, doc_id=doc_id)
    if not hits:
        return {"answer": "I couldn't find that in the document.", "sources": []}

    messages = list(history or [])
    messages.append(
        {
            "role": "user",
            "content": f"Sources:\n{build_context(hits)}\n\nQuestion: {question}",
        }
    )
    #msg = llm.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM, messages=messages)
    #return {"answer": msg.content[0].text, "sources": hits}
    resp = llm.chat.completions.create(
    model=MODEL,
    max_tokens=1000,
    messages=[{"role": "system", "content": SYSTEM}, *messages],
    )
    return {"answer": resp.choices[0].message.content, "sources": hits}