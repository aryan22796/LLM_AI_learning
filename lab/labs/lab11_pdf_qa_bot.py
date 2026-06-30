"""
Lab 11: PDF Q&A Bot
===================
Topic from syllabus:
RAG architecture, text chunking, FAISS/ChromaDB, similarity search, top-k, LLM synthesis.

What you learn:
1. Load a document.
2. Split text into chunks.
3. Create Gemini embeddings for chunks.
4. Search the most relevant chunks for a user question.
5. Send retrieved context to Gemini for final answer.

This file works with data/sample_document.txt by default.
If you want PDF support, install pypdf and set PDF_PATH to your PDF.

Run:
    python labs/lab11_pdf_qa_bot.py
"""

import numpy as np
from pathlib import Path
from gemini_client import ask_gemini, embed_text

TXT_PATH = Path("data/sample_document.txt")
PDF_PATH = None  # Example: Path("data/my_file.pdf")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    """Read PDF text using pypdf. This is optional."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 80):
    """
    Split long text into overlapping chunks.

    Why overlap?
    Important information may sit at a boundary between two chunks.
    Overlap prevents losing context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(question: str, chunks, chunk_embeddings, k: int = 3):
    q_emb = embed_text(question)
    scored = []
    for idx, emb in enumerate(chunk_embeddings):
        scored.append((idx, cosine_similarity(q_emb, emb)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def answer_question(question: str, chunks, top_matches):
    context = "\n\n".join([chunks[i] for i, score in top_matches])
    prompt = f"""
    You are a PDF Q&A assistant.
    Answer only using the provided context.
    If answer is not in context, say you do not know.

    Context:
    {context}

    Question:
    {question}

    Answer in simple language.
    """
    return ask_gemini(prompt)


def main():
    print("\n===== Lab 11: PDF Q&A Bot / RAG Bot =====\n")

    if PDF_PATH:
        raw_text = read_pdf(PDF_PATH)
    else:
        raw_text = read_text_file(TXT_PATH)

    chunks = chunk_text(raw_text)
    print(f"Created {len(chunks)} chunks")

    # Create embeddings for each chunk.
    # In production, save these embeddings in FAISS, ChromaDB, Vertex AI Vector Search, etc.
    chunk_embeddings = [embed_text(chunk) for chunk in chunks]

    question = "What is RAG and why do we use chunking?"
    top_matches = retrieve_top_k(question, chunks, chunk_embeddings, k=3)

    print("\nTop Retrieved Chunks:")
    for idx, score in top_matches:
        print(f"Chunk {idx} | score={score:.4f}")
        print(chunks[idx])
        print("-" * 60)

    answer = answer_question(question, chunks, top_matches)
    print("\nGemini Final Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
