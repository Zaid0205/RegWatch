# AI Governance & Compliance Assistant

A RAG-powered agent that ingests AI regulations (starting with the EU AI Act)
and classifies a described AI use case against the regulation's risk tiers,
citing the specific articles that justify the classification.

## Why this exists

Built as a one-week project to demonstrate applied RAG + agentic reasoning
in a policy/compliance context — a track that's increasingly relevant as
AI governance frameworks (EU AI Act, NIST AI RMF, etc.) go into force.

## Architecture

```
data/eu_ai_act.txt  --ingest.py-->  ChromaDB (persistent, local)
                                          |
user use-case description  ------>  risk_classifier.py --> Gemini 2.0 Flash --> structured JSON
                                          |
                                     app.py (FastAPI) --> /assess endpoint
```

- **Retrieval**: Sentence-Transformers (`all-MiniLM-L6-v2`) embeddings,
  stored in a local Chroma collection, chunked with article-level
  metadata tagging so citations are traceable back to source.
- **Reasoning**: Gemini is prompted to reason *only* from retrieved
  excerpts and return strict JSON (risk tier, confidence, reasoning,
  cited articles, compliance obligations, flags for human review).
  This keeps it honest and auditable rather than free-associating from
  pretraining knowledge of the regulation.

## Setup

1. Get a copy of the regulation text. For a first pass, source the
   consolidated EU AI Act text (official EUR-Lex publication) as a
   `.txt` or `.pdf` and place it at `data/eu_ai_act.txt`.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Build the vector store:
   ```
   python ingest.py --source data/eu_ai_act.txt --collection eu_ai_act
   ```
4. Set your Gemini API key:
   ```
   export GEMINI_API_KEY=your_key_here
   ```
5. Try it from the CLI:
   ```
   python risk_classifier.py --collection eu_ai_act \
     --use-case "An AI system that screens job applicant resumes and ranks candidates for interview."
   ```
6. Or run the API for a live demo:
   ```
   uvicorn app:app --reload
   ```

## Extending it

- Add more regulations by re-running `ingest.py` with a different
  `--collection` name (e.g. `nist_ai_rmf`), then pass that collection
  name in requests to compare how the same use case is treated under
  different frameworks — a strong differentiator for a demo video.
- Swap in a simple Streamlit or React front end over the `/assess`
  endpoint for a polished, screenshot-able UI.

## Disclaimer

This is a portfolio/demo project, not legal advice. Outputs should be
reviewed by a qualified compliance professional before any real-world
use.
