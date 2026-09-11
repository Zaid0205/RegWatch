"""
ingest.py — Build the RAG knowledge base for the AI Governance & Compliance Assistant.

What it does:
1. Loads regulation source documents (plain text / PDF) from ./data
2. Splits them into overlapping chunks (preserving article/section metadata)
3. Embeds chunks with Sentence-Transformers
4. Persists them into a local ChromaDB collection

Usage:
    python ingest.py --source data/eu_ai_act.txt --collection eu_ai_act

Source text prep:
- Drop a plain-text or PDF copy of the regulation into ./data
  (e.g. the official EU AI Act consolidated text, or a compliance
  guide PDF). For PDFs, this script uses pypdf to extract text.
- You can run this once per regulation and pass a different
  --collection name for each (e.g. "eu_ai_act", "nist_ai_rmf"),
  so the assistant can later be extended to multiple frameworks.
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


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_index(source_path: str, collection_name: str) -> int:
    """Programmatic entry point (used by ingest.py's CLI and by app.py's
    startup auto-build) — builds the collection and returns its final chunk count."""
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
            "article": tag_article_number(chunk) or "unspecified",
        })

    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.add(ids=ids[i:i + BATCH], documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH])

    return collection.count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to a .txt or .pdf regulation file")
    parser.add_argument("--collection", required=True, help="Name for this regulation's vector collection")
    args = parser.parse_args()

    print(f"Loading {args.source} ...")
    count = build_index(args.source, args.collection)
    print(f"Done. Collection '{args.collection}' now has {count} chunks.")


if __name__ == "__main__":
    main()
