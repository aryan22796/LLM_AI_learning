"""
Lab 11: PDF Q&A Bot using RAG
Topic: RAG architecture, text chunking, FAISS, similarity search, top-k retrieval, LLM synthesis.

Before running:
1. Put a PDF at data/sample.pdf
2. Run: python labs/lab11_pdf_qa_bot.py
"""

from pathlib import Path
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from utils.gemini_client import ask_gemini


PDF_PATH = Path("data/sample.pdf")


def read_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError("Put your PDF file at data/sample.pdf")

    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve(question, chunks, index, embedding_model, top_k=3):
    q_emb = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]


def answer_question(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful PDF question-answering assistant.
Answer the question using only the context below.
If answer is not in context, say you don't know from the PDF.

Context:
{context}

Question:
{question}
"""
    return ask_gemini(prompt)


def main():
    print("Reading PDF...")
    text = read_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Building FAISS index...")
    index, _ = build_faiss_index(chunks, embedding_model)

    print("\nPDF Q&A Bot ready. Type 'exit' to stop.")
    while True:
        question = input("\nAsk question: ")
        if question.lower() == "exit":
            break

        context_chunks = retrieve(question, chunks, index, embedding_model)
        answer = answer_question(question, context_chunks)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
