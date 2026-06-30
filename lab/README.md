# Gemini API Labs 8 to 12 - VS Code Project

This ZIP is for learning the topics shown in your syllabus using Gemini API in VS Code.

## Labs included

| Lab | Topic | File |
|---|---|---|
| Lab 8 | NLP Pipeline: TF-IDF + Logistic Regression + Gemini explanation | `labs/lab08_nlp_pipeline.py` |
| Lab 9 | LLM API Explorer: tokens, embeddings, cosine similarity | `labs/lab09_llm_api_explorer.py` |
| Lab 10 | Prompt Library: zero-shot, templates, JSON output, hallucination mitigation | `labs/lab10_prompt_library.py` |
| Lab 11 | PDF Q&A Bot / RAG: chunking, embeddings, similarity search, answer synthesis | `labs/lab11_pdf_qa_bot.py` |
| Lab 12 | RAG Evaluator: chunk-size tuning, relevance score, faithfulness check | `labs/lab12_rag_evaluator.py` |

## Setup in VS Code

### 1. Extract ZIP
Open the extracted folder in VS Code.

### 2. Create virtual environment

```bash
python -m venv env
```

Activate it:

Windows:

```bash
env\Scripts\activate
```

Mac/Linux:

```bash
source env/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Create .env file
Copy `.env.example` and rename it to `.env`.

Inside `.env`, paste your Gemini key:

```text
GENAI_API_KEY=your_actual_key_here
```

### 5. Run lab menu

```bash
python app.py
```

Then enter lab number: 8, 9, 10, 11, or 12.

## Run individual lab

```bash
python labs/lab08_nlp_pipeline.py
python labs/lab09_llm_api_explorer.py
python labs/lab10_prompt_library.py
python labs/lab11_pdf_qa_bot.py
python labs/lab12_rag_evaluator.py
```

## Important security note
Never upload `.env` to GitHub because it contains your API key.

## Learning order

1. Lab 8: Learn traditional NLP pipeline.
2. Lab 9: Understand LLM API and embeddings.
3. Lab 10: Learn prompt engineering patterns.
4. Lab 11: Build simple RAG bot.
5. Lab 12: Evaluate RAG quality.
