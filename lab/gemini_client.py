"""
gemini_client.py
----------------
Small wrapper around the Google GenAI SDK.

Why wrapper?
If every file creates the client separately, code becomes repeated.
A wrapper gives us one place to control model name, API key, and helper functions.
"""

from google import genai
from config import GENAI_API_KEY, DEFAULT_MODEL, EMBEDDING_MODEL

client = genai.Client(api_key=GENAI_API_KEY)


def ask_gemini(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to Gemini and return text response."""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def embed_text(text: str):
    """
    Create embedding vector using Gemini embedding model.

    Embedding = numeric representation of text.
    Similar meanings should have vectors close to each other.
    """
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return result.embeddings[0].values
