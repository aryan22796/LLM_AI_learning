import os
from dotenv import load_dotenv
from google import genai


def get_client():
    """Create Gemini client using API key from .env."""
    load_dotenv()
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY not found. Create .env and add your key.")
    return genai.Client(api_key=api_key)


def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Send simple prompt to Gemini and return text."""
    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text
