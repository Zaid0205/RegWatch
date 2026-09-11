"""
app.py — FastAPI demo endpoint for the AI Governance & Compliance Assistant.

Run:
    export GEMINI_API_KEY=...
    uvicorn app:app --reload

Then POST to /assess:
    {
      "use_case": "An AI system that screens job applicant resumes...",
      "collection": "eu_ai_act"
    }
"""

import os

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel

from risk_classifier import classify, format_human_readable, CHROMA_DIR, EMBED_MODEL
from ingest import build_index

app = FastAPI(title="AI Governance & Compliance Assistant")

DEFAULT_COLLECTION = "eu-ai-act"
SOURCE_PDF = "data/eu_ai_act.pdf"


@app.on_event("startup")
def ensure_index_built():
    """On a fresh server (e.g. after deployment), the vector database won't
    exist yet. This checks for it and builds it automatically from the PDF
    bundled in the repo, so no manual ingest.py step is needed on the host."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION, embedding_function=embed_fn)
        if collection.count() > 0:
            print(f"Knowledge base already built ({collection.count()} chunks). Skipping ingestion.")
            return
    except Exception:
        pass  # collection doesn't exist yet — fall through to build it

    if os.path.exists(SOURCE_PDF):
        print("Building knowledge base for the first time — this runs once per fresh deploy...")
        count = build_index(SOURCE_PDF, DEFAULT_COLLECTION)
        print(f"Knowledge base ready: {count} chunks indexed.")
    else:
        print(f"WARNING: {SOURCE_PDF} not found — /assess will fail until data is ingested.")


class AssessRequest(BaseModel):
    use_case: str
    collection: str = "eu-ai-act"


SIMPLE_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>AI Governance & Compliance Assistant</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
    textarea { width: 100%; height: 100px; font-size: 15px; padding: 8px; }
    button { padding: 10px 20px; font-size: 15px; margin-top: 10px; cursor: pointer; }
    pre { background: #f4f4f4; padding: 15px; white-space: pre-wrap; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>AI Governance & Compliance Assistant</h2>
  <p>Describe an AI system in plain English, and it will tell you its risk tier under the EU AI Act, with citations.</p>
  <textarea id="useCase" placeholder="e.g. An AI system that screens job applicant resumes and ranks candidates for interview."></textarea>
  <br>
  <button onclick="assess()">Check risk tier</button>
  <pre id="result"></pre>

  <script>
    async function assess() {
      const useCase = document.getElementById('useCase').value;
      const resultBox = document.getElementById('result');
      resultBox.textContent = "Thinking...";
      const response = await fetch('/assess-readable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_case: useCase, collection: 'eu-ai-act' })
      });
      const text = await response.text();
      resultBox.textContent = text;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    """A simple webpage with a text box — no JSON typing required."""
    return SIMPLE_PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess")
def assess(req: AssessRequest):
    """Returns the structured JSON result — for other programs/scripts to consume."""
    if not req.use_case.strip():
        raise HTTPException(status_code=400, detail="use_case cannot be empty")
    result = classify(req.use_case, req.collection)
    return result


@app.post("/assess-readable", response_class=PlainTextResponse)
def assess_readable(req: AssessRequest):
    """Returns a plain-English summary — use this one for demos and screenshots."""
    if not req.use_case.strip():
        raise HTTPException(status_code=400, detail="use_case cannot be empty")
    result = classify(req.use_case, req.collection)
    return format_human_readable(result)
