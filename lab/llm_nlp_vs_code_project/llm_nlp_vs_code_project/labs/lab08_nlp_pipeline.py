"""
Lab 8: NLP Pipeline
Topic: Feature selection, encoding, scaling, TF-IDF, word embeddings, BERT intro, sentiment analysis.
Project: Feature pipeline + TF-IDF + Logistic Regression on sample clinical-style notes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# Small teaching dataset. Replace with your real dataset later.
data = [
    ("patient reports chest pain and shortness of breath", "urgent"),
    ("mild cough and fever for two days", "normal"),
    ("severe headache with blurred vision", "urgent"),
    ("routine follow up patient feels better", "normal"),
    ("blood pressure extremely high and dizziness", "urgent"),
    ("patient came for regular diabetes checkup", "normal"),
    ("sudden weakness in left arm and slurred speech", "urgent"),
    ("minor throat pain no breathing issue", "normal"),
]


def main():
    df = pd.DataFrame(data, columns=["note", "label"])
    print("Dataset:")
    print(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["note"], df["label"], test_size=0.25, random_state=42
    )

    # Pipeline = TF-IDF feature extraction + Logistic Regression classifier
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    custom_note = ["patient has chest pain and dizziness"]
    print("\nCustom Prediction:")
    print(custom_note[0], "=>", model.predict(custom_note)[0])


if __name__ == "__main__":
    main()
