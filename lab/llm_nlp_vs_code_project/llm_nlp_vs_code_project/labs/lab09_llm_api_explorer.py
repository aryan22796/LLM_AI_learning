"""
Lab 9: LLM API Explorer
Topic: LLM internals, tokens, temperature, context window, embedding geometry, cosine similarity.
Project: Gemini API call + simple embedding similarity demo.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.gemini_client import ask_gemini


def llm_basic_call():
    prompt = "Explain tokens, temperature, and context window in simple words."
    answer = ask_gemini(prompt)
    print("\nGemini Response:\n")
    print(answer)


def similarity_demo():
    sentences = [
        "LLMs generate text by predicting the next token.",
        "Language models produce words one token at a time.",
        "A cricket match was played yesterday.",
        "Embeddings convert text into vectors.",
    ]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(vectors)

    print("\nCosine Similarity Matrix:")
    print(np.round(sim_matrix, 2))

    print("\nMost similar to sentence 0:")
    scores = sim_matrix[0]
    for i, score in enumerate(scores):
        print(i, score, "=>", sentences[i])


def main():
    llm_basic_call()
    similarity_demo()


if __name__ == "__main__":
    main()
