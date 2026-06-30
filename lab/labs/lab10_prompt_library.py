"""
Lab 10: Prompt Library
======================
Topic from syllabus:
Zero-shot, few-shot, chain-of-thought, system prompts, JSON mode, hallucination mitigation.

What you learn:
1. How to write reusable prompts.
2. Difference between zero-shot and few-shot prompting.
3. How to force structured JSON-style output.
4. How to reduce hallucination by giving context and constraints.

Run:
    python labs/lab10_prompt_library.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pathlib import Path
from gemini_client import ask_gemini


Base_DIR = Path(__file__).parent.parent
PROMPT_DIR = Base_DIR / "prompts" 


def load_prompt(name: str) -> str:
    """Load prompt template from prompts folder."""
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def main():
    print("\n===== Lab 10: Prompt Library =====\n")

    input_text = "RAG retrieves relevant document chunks before asking an LLM to answer."

    # Example 1: Zero-shot prompt.
    zero_shot = f"Explain this topic to a beginner: {input_text}"
    print("\n--- Zero-shot Output ---")
    print(ask_gemini(zero_shot))

    # Example 2: Template prompt from file.
    summarize_template = load_prompt("summarize.txt")
    prompt = summarize_template.format(input_text=input_text)
    print("\n--- Template Summary Output ---")
    print(ask_gemini(prompt))

    # Example 3: JSON-style extractor.
    json_template = load_prompt("json_extractor.txt")
    prompt = json_template.format(input_text=input_text)
    print("\n--- JSON Structured Output ---")
    print(ask_gemini(prompt))

    # Example 4: Hallucination mitigation.
    grounded_prompt = f"""
    Answer only using the given context.
    If the answer is not present, say: I do not know from the provided context.

    Context:
    {input_text}

    Question:
    What does RAG retrieve before answering?
    """
    print("\n--- Grounded Prompt Output ---")
    print(ask_gemini(grounded_prompt))


if __name__ == "__main__":
    main()
