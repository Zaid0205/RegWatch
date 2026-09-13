"""
graph.py — LangGraph orchestration for the AI Governance & Compliance Assistant.

Day 2: split the Day 1 monolithic compliance_agent_node into two independent
specialist agents that run IN PARALLEL off the same retrieved context:

  - risk_classifier_node -> risk_tier, confidence, reasoning
  - extractor_node        -> cited_articles, compliance_obligations,
                              flags_for_human_review

Why split instead of just two sequential calls: each agent now has a single
narrow job, which means (a) each can be evaluated independently in Day 5's
eval harness instead of scoring one blob of output, (b) a bad answer from one
doesn't force a full re-run of the other, and (c) since they're independent
of each other (both only depend on retrieve's output), they can run
concurrently instead of back-to-back — real latency reduction, not just a
code-organization change.

Graph shape now:

    START -> supervisor -> retrieve -> [risk_classifier, extractor] (parallel)
                                              \\        /
                                               finalize -> END

finalize_node is intentionally a thin pass-through today. Day 3 replaces it
with a citation_verifier_node that checks extractor's cited_articles against
retrieve's hits before the graph reaches END, and can loop back to extractor
on a failed citation.

Run directly:
    export GEMINI_API_KEY=...
    python graph.py --use-case "An AI system that screens resumes..."
"""

import argparse
import asyncio
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
GEMINI_MODEL = "gemini-3.6-flash"

RISK_CLASSIFIER_PROMPT = """You are a risk-tier classifier for AI systems under the EU AI Act.

You will be given a description of an AI system / use case and retrieved
excerpts from the regulation. Classify ONLY the risk tier — do not extract
obligations or citations, another agent handles that.

Base your answer only on the retrieved excerpts. If they don't clearly
support a tier, say "unclear" rather than guessing.

Return ONLY valid JSON, no markdown fences, no preamble:
{
  "risk_tier": "unacceptable" | "high-risk" | "limited-risk" | "minimal-risk" | "unclear",
  "confidence": "low" | "medium" | "high",
  "reasoning": "2-4 sentence explanation grounded in the excerpts"
}
"""

EXTRACTOR_PROMPT = """You are a compliance-obligations extractor for AI systems under the EU AI Act.

You will be given a description of an AI system / use case and retrieved
excerpts from the regulation. Extract citations and obligations ONLY — do
not classify a risk tier, another agent handles that.

Base your answer only on the retrieved excerpts. Only cite an article if it
is actually present in the excerpts.

Return ONLY valid JSON, no markdown fences, no preamble:
{
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
    # Set by risk_classifier_node
    risk_tier: Optional[str]
    confidence: Optional[str]
    reasoning: Optional[str]
    risk_error: Optional[str]
    risk_raw_output: Optional[str]
    # Set by extractor_node
    cited_articles: list[str]
    compliance_obligations: list[str]
    flags_for_human_review: list[str]
    extractor_error: Optional[str]
    extractor_raw_output: Optional[str]
    # Set by citation_verifier_node once it exists (Day 3)
    citations_verified: Optional[bool]
    # Set by finalize_node (consolidated error, if either agent failed)
    error: Optional[str]


def supervisor_node(state: GraphState) -> GraphState:
    """Entry point. Validates input today; becomes the real router once
    there's request-dependent branching to do (e.g. skip risk classification
    if the caller only wants obligations extracted)."""
    if not state.get("use_case", "").strip():
        return {**state, "error": "use_case cannot be empty"}
    return state


def retrieve_node(state: GraphState) -> GraphState:
    """Unchanged from Day 1."""
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


def _build_user_prompt(state: GraphState) -> str:
    return f"""AI USE CASE:
{state['use_case']}

RETRIEVED EXCERPTS:
{state['context']}
"""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    return raw


async def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Runs the (synchronous) Gemini SDK call in a thread so risk_classifier_node
    and extractor_node can genuinely run concurrently under LangGraph's async
    execution instead of blocking each other."""

    def _sync_call() -> str:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
        response = model.generate_content(user_prompt)
        return response.text

    return await asyncio.to_thread(_sync_call)


async def risk_classifier_node(state: GraphState) -> GraphState:
    raw = await _call_gemini(RISK_CLASSIFIER_PROMPT, _build_user_prompt(state))
    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"risk_error": "Risk classifier did not return valid JSON", "risk_raw_output": raw}
    return parsed


async def extractor_node(state: GraphState) -> GraphState:
    raw = await _call_gemini(EXTRACTOR_PROMPT, _build_user_prompt(state))
    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"extractor_error": "Extractor did not return valid JSON", "extractor_raw_output": raw}
    return parsed


def finalize_node(state: GraphState) -> GraphState:
    """Fan-in point after the two parallel agents. Today just consolidates
    errors from either branch into one field. Day 3 replaces this with a
    citation_verifier_node that actually checks extractor output against
    retrieve's hits."""
    errors = []
    if state.get("risk_error"):
        errors.append(f"Risk classifier: {state['risk_error']}")
    if state.get("extractor_error"):
        errors.append(f"Extractor: {state['extractor_error']}")
    if errors:
        return {**state, "error": "; ".join(errors)}
    return state


def route_after_supervisor(state: GraphState) -> str:
    return END if state.get("error") else "retrieve"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("risk_classifier", risk_classifier_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {"retrieve": "retrieve", END: END})
    # Fan-out: both agents depend only on retrieve's output, so they run in
    # the same superstep, concurrently, under ainvoke().
    graph.add_edge("retrieve", "risk_classifier")
    graph.add_edge("retrieve", "extractor")
    # Fan-in: finalize waits for both before the graph proceeds.
    graph.add_edge("risk_classifier", "finalize")
    graph.add_edge("extractor", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


async def run_async(use_case: str, collection: str = "eu-ai-act") -> dict:
    app_graph = build_graph()
    return await app_graph.ainvoke({"use_case": use_case, "collection": collection})


def run(use_case: str, collection: str = "eu-ai-act") -> dict:
    """Sync wrapper — app.py can keep calling run() without becoming async."""
    return asyncio.run(run_async(use_case, collection))


def format_human_readable(result: dict) -> str:
    if result.get("error") and not (result.get("risk_tier") or result.get("cited_articles")):
        raw = result.get("risk_raw_output") or result.get("extractor_raw_output") or ""
        return f"Something went wrong: {result['error']}\n\nRaw output:\n{raw}"

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
    if result.get("error"):
        lines += [f"(Note: {result['error']} — some fields above may be incomplete)"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--collection", default="eu-ai-act")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(args.use_case, args.collection)
    print(json.dumps(result, indent=2) if args.json else format_human_readable(result))
