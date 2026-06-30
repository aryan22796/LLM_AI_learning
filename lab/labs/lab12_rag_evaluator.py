"""
Lab 12: RAG Evaluator
=====================
Topic from syllabus:
Chunk-size tuning, metadata filtering, HYDE, re-ranking, faithfulness, relevance metrics.

What you learn:
1. Why RAG needs evaluation.
2. How to compare chunk sizes.
3. How to measure retrieval relevance using similarity score.
4. How Gemini can judge answer faithfulness against context.

Run:
    python labs/lab12_rag_evaluator.py
"""

import numpy as np
from pathlib import Path
from gemini_client import ask_gemini, embed_text

DOC_PATH = Path("data/sample_document.txt")


def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_retrieval(question: str, chunks):
    """Return best chunk and similarity score."""
    q_emb = embed_text(question)
    scores = []
    for i, chunk in enumerate(chunks):
        scores.append((i, cosine_similarity(q_emb, embed_text(chunk))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0]


def generate_answer(question: str, context: str):
    prompt = f"""
    Answer the question using only this context.

    Context:
    {context}

    Question:
    {question}
    """
    return ask_gemini(prompt)


def judge_faithfulness(question: str, context: str, answer: str):
    """
    Use Gemini as an evaluator.
    Faithfulness means the answer is supported by the retrieved context.
    """
    prompt = f"""
    You are a strict RAG evaluator.
    Check whether the answer is fully supported by the context.
    Return JSON only:
    {{"faithfulness_score": 0-5, "reason": "short reason"}}

    Context:
    {context}

    Question:
    {question}

    Answer:
    {answer}
    """
    return ask_gemini(prompt)


def main():
    print("\n===== Lab 12: RAG Evaluator =====\n")
    text = DOC_PATH.read_text(encoding="utf-8")
    question = "Why is evaluation important in RAG?"

    configs = [
        {"chunk_size": 150, "overlap": 30},
        {"chunk_size": 300, "overlap": 60},
        {"chunk_size": 500, "overlap": 100},
    ]

    print("Testing different chunk sizes:\n")
    for cfg in configs:
        chunks = chunk_text(text, cfg["chunk_size"], cfg["overlap"])
        best_idx, best_score = evaluate_retrieval(question, chunks)
        print(f"chunk_size={cfg['chunk_size']} overlap={cfg['overlap']} chunks={len(chunks)} best_score={best_score:.4f}")

    # Use one configuration for final answer.
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    best_idx, best_score = evaluate_retrieval(question, chunks)
    context = chunks[best_idx]
    answer = generate_answer(question, context)

    print("\nBest Retrieved Context:\n")
    print(context)
    print("\nGenerated Answer:\n")
    print(answer)

    print("\nFaithfulness Evaluation:\n")
    print(judge_faithfulness(question, context, answer))

    print("\nLearning point:")
    print("Good RAG is not only generation. You must evaluate retrieval relevance and answer faithfulness.")


if __name__ == "__main__":
    main()
