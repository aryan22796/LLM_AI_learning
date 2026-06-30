"""
Lab 10: Prompt Library
Topic: Zero-shot, few-shot, chain-of-thought style prompting, system prompts, JSON mode, hallucination mitigation.
Project: Reusable prompt templates with structured output.
"""

from utils.gemini_client import ask_gemini


def zero_shot():
    prompt = "Classify this review as positive, negative, or neutral: The product is okay, not great."
    return ask_gemini(prompt)


def few_shot():
    prompt = """
You classify sentiment.

Example 1:
Text: I love this product.
Sentiment: positive

Example 2:
Text: This is terrible.
Sentiment: negative

Now classify:
Text: Delivery was late but quality is good.
Sentiment:
"""
    return ask_gemini(prompt)


def json_output():
    prompt = """
Extract information from this text and return only valid JSON.

Text: Nagendra is learning LLMs on GCP using VS Code.

Required JSON fields:
- name
- topic
- platform
- tool
"""
    return ask_gemini(prompt)


def hallucination_safe_prompt():
    prompt = """
Answer the question using only the given context.
If the answer is not present, say: I don't know from the provided context.

Context:
Vertex AI is a Google Cloud platform for building and using AI models.

Question:
Does Vertex AI support building AI applications?
"""
    return ask_gemini(prompt)


def main():
    print("\nZERO SHOT:\n", zero_shot())
    print("\nFEW SHOT:\n", few_shot())
    print("\nJSON OUTPUT:\n", json_output())
    print("\nHALLUCINATION SAFE:\n", hallucination_safe_prompt())


if __name__ == "__main__":
    main()
