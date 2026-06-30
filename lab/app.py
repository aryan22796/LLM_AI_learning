"""
app.py
------
Main menu to run labs 8 to 12.

Run:
    python app.py
"""

import subprocess
import sys

LABS = {
    "8": "labs/lab08_nlp_pipeline.py",
    "9": "labs/lab09_llm_api_explorer.py",
    "10": "labs/lab10_prompt_library.py",
    "11": "labs/lab11_pdf_qa_bot.py",
    "12": "labs/lab12_rag_evaluator.py",
}

print("Gemini LLM Labs 8 to 12")
print("8  - NLP Pipeline")
print("9  - LLM API Explorer")
print("10 - Prompt Library")
print("11 - PDF Q&A / RAG Bot")
print("12 - RAG Evaluator")

choice = input("\nEnter lab number: ").strip()

if choice not in LABS:
    print("Invalid choice")
    sys.exit(1)

subprocess.run([sys.executable, LABS[choice]])
