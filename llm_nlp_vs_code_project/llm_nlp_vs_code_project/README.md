# LLM + NLP VS Code Project

This project covers Labs 8–11:

- Lab 8: NLP Pipeline with TF-IDF + Logistic Regression
- Lab 9: LLM API Explorer + token concepts + embedding similarity
- Lab 10: Prompt Library with zero-shot, few-shot, JSON output
- Lab 11: PDF Q&A Bot using RAG with FAISS

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Create `.env`:

```bash
GENAI_API_KEY=your_api_key_here
```

## Run files

```bash
python labs/lab08_nlp_pipeline.py
python labs/lab09_llm_api_explorer.py
python labs/lab10_prompt_library.py
python labs/lab11_pdf_qa_bot.py
```

Put your PDF file in `data/sample.pdf` before running Lab 11.
