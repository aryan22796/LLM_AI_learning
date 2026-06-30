"""
Lab 9: LLM API Explorer
=======================
Topic from syllabus:
LLM internals, tokens, temperature, context window, embedding geometry, cosine/dot similarity.

What you learn:
1. How to call Gemini from Python.
2. What temperature means.
3. How embeddings represent meaning.
4. How cosine similarity compares two text vectors.

Run:
    python labs/lab09_llm_api_explorer.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from gemini_client import ask_gemini, embed_text


def cosine_similarity(a, b):
    """Cosine similarity = angle similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("\n===== Lab 9: LLM API Explorer =====\n")

    prompt = "Explain tokens in LLMs using a simple food example."
    print("Prompt:", prompt)
    print("\nGemini Response:\n")
    print(ask_gemini(prompt))

    # Embedding similarity example.
    text1 = "A patient has fever and cough."
    text2 = "The person is sick with flu symptoms."
    text3 = "A car engine needs maintenance."

    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    emb3 = embed_text(text3)

    print("\nCosine Similarity Scores:")
    print("text1 vs text2:", cosine_similarity(emb1, emb2))
    print("text1 vs text3:", cosine_similarity(emb1, emb3))

    print("\nLearning point:")
    print("Higher similarity means texts are semantically closer.")


if __name__ == "__main__":
    main()
