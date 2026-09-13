"""
graph.py — LangGraph orchestration for the AI Governance & Compliance Assistant.

Day 3: add a Citation-Verifier agent after the Extractor. It checks every
article the Extractor cited against what was actually retrieved (state["hits"]),
since the Extractor and Risk Classifier no longer see each other's output and
could otherwise cite something that isn't really in the retrieved context.

If a citation can't be traced back to a retrieved chunk, the graph loops back
to the Extractor for one more attempt (bounded — MAX_EXTRACTOR_ATTEMPTS caps
it so a persistently wrong model can't loop forever). If it still can't
verify after retries, it flags the citation for human review instead of
silently returning it.

Graph shape now:

    START -> supervisor -> retrieve -> risk_classifier ---------\\
                                    \\-> extractor -> citation_verifier -> finalize -> END
                                                          ^              /
                                                          \\--(retry)---/

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
MAX_EXTRACTOR_ATTEMPTS = 2  # total attempts allowed: 1 initial + (MAX_EXTRACTOR_ATTEMPTS - 1) retries

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

Base your answer only on the retrieved excerpts. ONLY cite an article if its
exact label appears in the excerpts below — do not cite from general
knowledge of the EU AI Act.

Return ONLY valid JSON, no markdown fences, no preamble:
{
  "cited_articles": ["Article X", "Article Y"],
  "compliance_obligations": ["short bullet", "short bullet"],
  "flags_for_human_review": ["anything ambiguous or missing from context"]
}
"""

EXTRACTOR_RETRY_NOTE = """
NOTE: Your previous answer cited article(s) that could not be found in the
retrieved excerpts: {bad_citations}. Only cite articles whose exact label
appears below. If you can't find support for a claim, list it under
flags_for_human_review instead of citing an unsupported article.
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
    # Set by citation_verifier_node
    citations_verified: Optional[bool]
    unverifiable_citations: list[str]
    extractor_attempts: int
    # Set by finalize_node
    error: Optional[str]


def supervisor_node(state: GraphState) -> GraphState:
    if not state.get("use_case", "").strip():
        return {**state, "error": "use_case cannot be empty"}
    return state


def retrieve_node(state: GraphState) -> GraphState:
    """Unchanged from Day 1/2."""
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


def _build_user_prompt(state: GraphState, retry_note: str = "") -> str:
    return f"""AI USE CASE:
{state['use_case']}
{retry_note}
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
    retry_note = ""
    if state.get("unverifiable_citations"):
        retry_note = EXTRACTOR_RETRY_NOTE.format(bad_citations=", ".join(state["unverifiable_citations"]))

    raw = await _call_gemini(EXTRACTOR_PROMPT, _build_user_prompt(state, retry_note))
    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"extractor_error": "Extractor did not return valid JSON", "extractor_raw_output": raw}
    # Clear any stale verification result from a previous attempt so the
    # verifier re-checks the fresh citations rather than reusing old state.
    return {**parsed, "citations_verified": None, "unverifiable_citations": []}


def citation_verifier_node(state: GraphState) -> GraphState:
    """Checks every cited article against what was actually retrieved.
    Uses substring matching in both directions (rather than exact equality)
    since a model might cite "Article 6" when the chunk metadata says
    "Article 6(1)", or vice versa."""
    hits = state.get("hits", [])
    valid_articles = [h["article"].lower() for h in hits]
    cited = state.get("cited_articles", [])

    unverifiable = [
        c for c in cited
        if not any(c.lower() in article or article in c.lower() for article in valid_articles)
    ]

    if not unverifiable:
        return {"citations_verified": True, "unverifiable_citations": []}

    return {
        "citations_verified": False,
        "unverifiable_citations": unverifiable,
        "extractor_attempts": state.get("extractor_attempts", 0) + 1,
    }


def route_after_verification(state: GraphState) -> str:
    if state.get("citations_verified"):
        return "finalize"
    if state.get("extractor_attempts", 0) >= MAX_EXTRACTOR_ATTEMPTS:
        # Give up retrying. Don't silently drop the bad citations — push them
        # into flags_for_human_review so a person sees them instead of them
        # vanishing.
        return "give_up"
    return "extractor"


def give_up_node(state: GraphState) -> GraphState:
    flags = list(state.get("flags_for_human_review", []))
    for c in state.get("unverifiable_citations", []):
        flags.append(f"Citation '{c}' could not be verified against retrieved excerpts after {MAX_EXTRACTOR_ATTEMPTS} attempts")
    remaining_cited = [c for c in state.get("cited_articles", []) if c not in state.get("unverifiable_citations", [])]
    return {"cited_articles": remaining_cited, "flags_for_human_review": flags, "citations_verified": False}


def finalize_node(state: GraphState) -> GraphState:
    """Fan-in point. IMPORTANT: only return the keys this node actually
    changes — returning the whole state here (e.g. {**state, ...}) makes
    LangGraph treat every field as a fresh write, which collides with
    citation_verifier writing to the same channel (citations_verified) in
    the same step and throws InvalidUpdateError. Each node in this graph
    follows the same rule: return only what you changed."""
    errors = []
    if state.get("risk_error"):
        errors.append(f"Risk classifier: {state['risk_error']}")
    if state.get("extractor_error"):
        errors.append(f"Extractor: {state['extractor_error']}")
    if errors:
        return {"error": "; ".join(errors)}
    return {}


def route_after_supervisor(state: GraphState) -> str:
    return END if state.get("error") else "retrieve"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("risk_classifier", risk_classifier_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("citation_verifier", citation_verifier_node)
    graph.add_node("give_up", give_up_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {"retrieve": "retrieve", END: END})

    graph.add_edge("retrieve", "risk_classifier")
    graph.add_edge("retrieve", "extractor")

    graph.add_edge("extractor", "citation_verifier")
    graph.add_conditional_edges(
        "citation_verifier",
        route_after_verification,
        {"finalize": "finalize", "give_up": "give_up", "extractor": "extractor"},
    )
    graph.add_edge("give_up", "finalize")

    graph.add_edge("risk_classifier", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


async def run_async(use_case: str, collection: str = "eu-ai-act") -> dict:
    app_graph = build_graph()
    return await app_graph.ainvoke({"use_case": use_case, "collection": collection})


def run(use_case: str, collection: str = "eu-ai-act") -> dict:
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
        lines += ["Legal basis cited (verified against retrieved text):"] + [f"  - {a}" for a in articles] + [""]
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
