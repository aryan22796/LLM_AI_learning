"""
Main starter app for Gemini in VS Code.
Run: python app.py
"""

from utils.gemini_client import ask_gemini


def main():
    print("Gemini Chatbot. Type 'exit' to stop.")
    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break
        print("\nGemini:")
        print(ask_gemini(question))


if __name__ == "__main__":
    main()
