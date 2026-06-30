"""
config.py
----------
Central configuration file for all labs.

Why this file exists:
In real projects, we do not hardcode API keys directly in Python files.
We keep keys in a .env file and load them using python-dotenv.

Steps:
1. Copy .env.example to .env
2. Paste your Gemini API key inside .env
3. Run any lab file
"""

import os
from dotenv import load_dotenv

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

if not GENAI_API_KEY:
    raise ValueError(
        "GENAI_API_KEY not found. Create a .env file and add: GENAI_API_KEY=your_key"
    )
