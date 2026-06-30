"""
Lab 8: NLP Pipeline
===================
Topic from syllabus:
Feature selection, encoding, scaling, TF-IDF, word embeddings, BERT intro, sentiment analysis.

What you learn in this lab:
1. How raw text becomes features.
2. How TF-IDF converts text into numbers.
3. How Logistic Regression can classify sentiment.
4. How Gemini can explain model predictions in human language.

Run:
    python labs/lab08_nlp_pipeline.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # add parent dir to path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gemini_client import ask_gemini

Base_DIR = Path(__file__).parent.parent
DATA_PATH = Base_DIR / "data" / "clinical_notes.csv"


def main():
    print("\n===== Lab 8: NLP Pipeline =====\n")

    # 1. Load small dataset.
    # In real clinical NLP, notes may come from EMR, PDF, database, or APIs.
    df = pd.read_csv(DATA_PATH)
    print("Dataset:")
    print(df)

    # 2. Split data.
    # train_test_split helps us test whether model generalizes to unseen text.
    X_train, X_test, y_train, y_test = train_test_split(
        df["note"], df["label"], test_size=0.33, random_state=42
    )

    # 3. Build pipeline.
    # TF-IDF: gives high score to important words and lower score to common words.
    # LogisticRegression: simple classification model.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", LogisticRegression())
    ])

    # 4. Train model.
    model.fit(X_train, y_train)

    # 5. Evaluate model.
    preds = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, preds, zero_division=0))

    # 6. Predict new note.
    new_note = "Patient has fever and severe weakness. Follow up required."
    prediction = model.predict([new_note])[0]
    print("\nNew Note:", new_note)
    print("Prediction:", prediction)

    # 7. Ask Gemini to explain the prediction.
    # This is how traditional ML + LLM can work together.
    prompt = f"""
    You are teaching NLP to beginners.
    A Logistic Regression sentiment model predicted this clinical note as: {prediction}

    Clinical note:
    {new_note}

    Explain why this prediction may happen using simple NLP terms like tokens and TF-IDF.
    """
    explanation = ask_gemini(prompt)
    print("\nGemini Explanation:\n")
    print(explanation)


if __name__ == "__main__":
    main()
