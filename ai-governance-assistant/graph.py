"""
graph.py — LangGraph orchestration for the AI Governance & Compliance Assistant.

Day 1 refactor: wraps the existing single-shot classifier (previously
risk_classifier.classify()) as one node in a LangGraph StateGraph, behind a
Supervisor entry point.

Today the graph is linear:
    START -> supervisor -> retrieve -> compliance_agent -> END

It's linear on purpose for Day 1 — the goal isn't new behavior yet, it's
correct seams. GraphState already carries every field the Day 2/3 agents
will need, so:
  - Day 2 splits compliance_agent into risk_classifier_node (risk_tier,
    confidence) + extractor_node (cited_articles, compliance_obligations) —
    both write into the same state dict, no schema changes needed.
  - Day 3 adds a citation_verifier_node after compliance_agent that checks
    cited_articles against hits (the actual retrieved chunks) before the
    graph reaches END, and can route back to compliance_agent on failure.
  - supervisor_node is intentionally trivial today (just input validation).
    From Day 2 on, it becomes the router deciding which specialist(s) to
    call for a given request (e.g. skip risk classification if the user
    only wants obligations extracted).

Run directly:
    export GEMINI_API_KEY=...
    python graph.py --use-case "An AI system that screens resumes..."
"""

import argparse
import json
import os
from typing import Optional, TypedDict

import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from langgraph.graph import END, StateGraph

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10

# Unchanged from risk_classifier.py — moving this wholesale for Day 1 so
# behavior stays identical while the execution shape changes underneath it.
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


class GraphState(TypedDict, total=False):
    # Input
    use_case: str
    collection: str
    # Set by retrieve_node
    hits: list[dict]
    context: str
    # Set by compliance_agent_node (Day 2 will split these across two nodes)
    risk_tier: Optional[str]
    confidence: Optional[str]
    reasoning: Optional[str]
    cited_articles: list[str]
    compliance_obligations: list[str]
    flags_for_human_review: list[str]
    # Set by citation_verifier_node once it exists (Day 3)
    citations_verified: Optional[bool]
    # Error path
    error: Optional[str]
    raw_output: Optional[str]


def supervisor_node(state: GraphState) -> GraphState:
    """Entry point. Validates input today; becomes the real router in Day 2+
    once there's more than one specialist to route between."""
    if not state.get("use_case", "").strip():
        return {**state, "error": "use_case cannot be empty"}
    return state


def retrieve_node(state: GraphState) -> GraphState:
    """Same retrieval logic as risk_classifier.retrieve() — just living as a
    graph node now so it's a reusable step other agents can call into."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name=state["collection"], embedding_function=embed_fn)

    results = collection.query(query_texts=[state["use_case"]], n_results=TOP_K)
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": doc, "article": meta.get("article", "unspecified")})

    context_lines = [f"[{h['article']}]\n{h['text'].strip()}\n" for h in hits]
    context = "\n---\n".join(context_lines)

    return {**state, "hits": hits, "context": context}


def compliance_agent_node(state: GraphState) -> GraphState:
    """Your existing classify() logic, now living as one graph node instead
    of a standalone function. Day 2 splits this into a Risk Classifier node
    and an Extractor node so each can be evaluated independently."""
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-3.6-flash", system_instruction=SYSTEM_PROMPT)

    prompt = f"""AI USE CASE:
{state['use_case']}

RETRIEVED EXCERPTS:
{state['context']}
"""
    response = model.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {**state, "error": "Model did not return valid JSON", "raw_output": raw}

    return {**state, **parsed}


def route_after_supervisor(state: GraphState) -> str:
    return END if state.get("error") else "retrieve"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compliance_agent", compliance_agent_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {"retrieve": "retrieve", END: END})
    graph.add_edge("retrieve", "compliance_agent")
    graph.add_edge("compliance_agent", END)

    return graph.compile()


def run(use_case: str, collection: str = "eu-ai-act") -> dict:
    app_graph = build_graph()
    return app_graph.invoke({"use_case": use_case, "collection": collection})


def format_human_readable(result: dict) -> str:
    """Unchanged from risk_classifier.py — app.py's /assess-readable route
    can keep importing this from either module during the transition."""
    if "error" in result:
        return f"Something went wrong: {result['error']}\n\nRaw output:\n{result.get('raw_output', '')}"

    tier = result.get("risk_tier", "unknown").upper()
    confidence = result.get("confidence", "unknown")
    reasoning = result.get("reasoning", "")
    articles = result.get("cited_articles", [])
    obligations = result.get("compliance_obligations", [])
    flags = result.get("flags_for_human_review", [])

    lines = ["=" * 60, f"RISK CLASSIFICATION: {tier}", f"Confidence: {confidence}", "=" * 60, "", "Why:", f"  {reasoning}", ""]
    if articles:
        lines += ["Legal basis cited:"] + [f"  - {a}" for a in articles] + [""]
    if obligations:
        lines += ["What you'd need to do if this is high-risk:"] + [f"  - {o}" for o in obligations] + [""]
    if flags:
        lines += ["Worth double-checking with a human/lawyer:"] + [f"  - {f}" for f in flags] + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--collection", default="eu-ai-act")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(args.use_case, args.collection)
    print(json.dumps(result, indent=2) if args.json else format_human_readable(result))
