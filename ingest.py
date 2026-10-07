"""
ingest.py — Build the RAG knowledge base for the AI Governance & Compliance Assistant.

Day 4: added tag_principle() so a second framework (SDAIA's AI Ethics
Principles, which is organized around seven named principles rather than
numbered articles) can be ingested into its own collection alongside the
existing EU AI Act one.

What it does:
1. Loads regulation source documents (plain text / PDF) from ./data
2. Splits them into overlapping chunks (preserving article/principle metadata)
3. Embeds chunks with Sentence-Transformers
4. Persists them into a local ChromaDB collection

Usage:
    python ingest.py --source data/eu_ai_act.pdf --collection eu-ai-act
    python ingest.py --source data/sdaia_ai_ethics.pdf --collection sdaia-ai-ethics --label-style principles

--label-style controls which citation-tagging function runs over each chunk:
  - "articles"   -> looks for "Article N" / "Annex N" / "Point X(y)" (EU AI Act style)
  - "principles" -> looks for SDAIA's seven named principles
  - "auto" (default) -> guesses from the --collection name (anything with
    "sdaia" in it uses principles, everything else uses articles)
"""

import argparse
import os
import re
import uuid

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1200          # characters per chunk
CHUNK_OVERLAP = 200        # characters of overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"  # matches your existing MiniLM usage

# SDAIA AI Ethics Principles 2.0's seven principles. Matched case-insensitively
# against chunk text, longest/most-specific patterns first so "Privacy and
# Security" doesn't get missed by a looser match.
SDAIA_PRINCIPLES = [
    "Social and Environmental Benefits",
    "Social & Environmental Benefits",
    "Accountability and Responsibility",
    "Accountability & Responsibility",
    "Transparency and Explainability",
    "Transparency & Explainability",
    "Privacy and Security",
    "Privacy & Security",
    "Reliability and Safety",
    "Reliability & Safety",
    "Fairness",
    "Humanity",
]


def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tag_article_number(chunk: str) -> str | None:
    """Best-effort extraction of a citation label so citations are traceable.

    Tries, in order: 'Article N', 'Annex N', and 'Point X(y)' — since the
    EU AI Act's high-risk classifications live in Annex III as numbered
    points, not just numbered Articles.
    """
    annex_point = re.search(r"Annex\s+([IVXLC]+).{0,80}?[Pp]oint\s+(\d+[a-z]?(?:\(\w\))?)", chunk, re.DOTALL)
    if annex_point:
        return f"Annex {annex_point.group(1)}, Point {annex_point.group(2)}"

    annex = re.search(r"Annex\s+([IVXLC]+)", chunk)
    if annex:
        return f"Annex {annex.group(1)}"

    article = re.search(r"Article\s+(\d+[a-z]?)", chunk)
    if article:
        return f"Article {article.group(1)}"

    return None


def tag_principle(chunk: str) -> str | None:
    """Best-effort extraction of which SDAIA principle a chunk falls under,
    for frameworks organized around named principles rather than numbered
    articles."""
    for principle in SDAIA_PRINCIPLES:
        if re.search(re.escape(principle), chunk, re.IGNORECASE):
            return f"Principle: {principle}"
    return None


def tag_citation(chunk: str, label_style: str) -> str | None:
    if label_style == "principles":
        return tag_principle(chunk)
    return tag_article_number(chunk)


def resolve_label_style(label_style: str, collection_name: str) -> str:
    if label_style != "auto":
        return label_style
    return "principles" if "sdaia" in collection_name.lower() else "articles"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_index(source_path: str, collection_name: str, label_style: str = "auto") -> int:
    """Programmatic entry point (used by ingest.py's CLI and by app.py's
    startup auto-build) — builds the collection and returns its final chunk count."""
    resolved_style = resolve_label_style(label_style, collection_name)

    raw_text = load_text(source_path)
    chunks = chunk_text(raw_text)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    ids, docs, metas = [], [], []
    for chunk in chunks:
        ids.append(str(uuid.uuid4()))
        docs.append(chunk)
        metas.append({
            "source": os.path.basename(source_path),
            "article": tag_citation(chunk, resolved_style) or "unspecified",
        })

    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.add(ids=ids[i:i + BATCH], documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH])

    return collection.count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to a .txt or .pdf regulation file")
    parser.add_argument("--collection", required=True, help="Name for this regulation's vector collection")
    parser.add_argument("--label-style", default="auto", choices=["auto", "articles", "principles"],
                         help="Citation-tagging strategy. 'auto' infers from --collection name.")
    args = parser.parse_args()

    print(f"Loading {args.source} ...")
    count = build_index(args.source, args.collection, args.label_style)
    print(f"Done. Collection '{args.collection}' now has {count} chunks.")


if __name__ == "__main__":
    main()
