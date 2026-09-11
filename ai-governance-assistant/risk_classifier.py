"""
risk_classifier.py — RAG + Gemini agent that classifies an AI use case
against the EU AI Act's risk tiers and cites the supporting articles.

Usage:
    export GEMINI_API_KEY=...
    python risk_classifier.py --collection eu_ai_act \
        --use-case "An AI system that screens job applicant resumes
                    and ranks candidates for interview."

Design notes:
- Retrieval: top-k chunks from Chroma, filtered to the most relevant
  articles for the query.
- The prompt forces STRICT JSON output so this can sit behind the
  FastAPI endpoint in app.py without brittle string parsing.
- Risk tiers follow the EU AI Act's four-tier structure:
  unacceptable / high-risk / limited-risk / minimal-risk.
"""

import argparse
import json
import os

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10

SYSTEM_PROMPT = """You are an AI regulatory compliance analyst. You will be given:
1. A description of an AI system / use case.
2. Retrieved excerpts from a regulation (with article labels).

Your job: classify the use case's risk tier under the EU AI Act and
justify it ONLY using the retrieved excerpts. If the excerpts do not
clearly support a tier, say so rather than guessing.

Return ONLY valid JSON, no markdown fences, no preamble, matching this schema:
{
  "risk_tier": "unacceptable" | "high-risk" | "limited-risk" | "minimal-risk" | "unclear",
  "confidence": "low" | "medium" | "high",
  "reasoning": "2-4 sentence explanation grounded in the excerpts",
  "cited_articles": ["Article X", "Article Y"],
  "compliance_obligations": ["short bullet", "short bullet"],
  "flags_for_human_review": ["anything ambiguous or missing from context"]
}
"""


def retrieve(collection_name: str, query: str, k: int = TOP_K):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name=collection_name, embedding_function=embed_fn)

    results = collection.query(query_texts=[query], n_results=k)
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": doc, "article": meta.get("article", "unspecified")})
    return hits


def build_context_block(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        lines.append(f"[{h['article']}]\n{h['text'].strip()}\n")
    return "\n---\n".join(lines)


def classify(use_case: str, collection_name: str) -> dict:
    hits = retrieve(collection_name, use_case)
    context = build_context_block(hits)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        "gemini-3.6-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    user_prompt = f"""AI USE CASE:
{use_case}

RETRIEVED EXCERPTS:
{context}
"""

    response = model.generate_content(user_prompt)
    raw = response.text.strip()

    # Defensive parsing in case the model wraps output in fences anyway
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Model did not return valid JSON", "raw_output": raw}


def format_human_readable(result: dict) -> str:
    """Render the classification result as plain, readable text instead of raw JSON."""
    if "error" in result:
        return f"Something went wrong: {result['error']}\n\nRaw output:\n{result.get('raw_output', '')}"

    tier = result.get("risk_tier", "unknown").upper()
    confidence = result.get("confidence", "unknown")
    reasoning = result.get("reasoning", "")
    articles = result.get("cited_articles", [])
    obligations = result.get("compliance_obligations", [])
    flags = result.get("flags_for_human_review", [])

    lines = []
    lines.append("=" * 60)
    lines.append(f"RISK CLASSIFICATION: {tier}")
    lines.append(f"Confidence: {confidence}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Why:")
    lines.append(f"  {reasoning}")
    lines.append("")

    if articles:
        lines.append("Legal basis cited:")
        for a in articles:
            lines.append(f"  - {a}")
        lines.append("")

    if obligations:
        lines.append("What you'd need to do if this is high-risk:")
        for o in obligations:
            lines.append(f"  - {o}")
        lines.append("")

    if flags:
        lines.append("Worth double-checking with a human/lawyer:")
        for f in flags:
            lines.append(f"  - {f}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of the readable summary")
    args = parser.parse_args()

    result = classify(args.use_case, args.collection)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_human_readable(result))


if __name__ == "__main__":
    main()
