"""
graph.py — LangGraph Workflow
Connects all nodes with conditional edges and routing logic.
See README.md for full architecture and flow diagram.
"""

from langgraph.graph import StateGraph, END
from state import RAGState

from nodes.query_analysis    import query_analysis_node
from nodes.retrieval         import retrieval_node
from nodes.grading           import grading_node
from nodes.generation        import generation_node
from nodes.hallucination     import hallucination_check_node
from nodes.web_search        import web_search_node

MAX_RETRIES = 2


# ── Routing Functions ──────────────────────────────────────────────────────────

def route_after_grading(state: RAGState) -> str:
    """
    Routes after grading based on:
    - relevant_docs found      → generation
    - no relevant_docs
        - used_web_search      → no more fallback → END
        - retry_count < MAX    → query_analysis (rewrite + retry)
        - retry_count >= MAX   → web_search
    """
    relevant_docs   = state.get("relevant_docs", [])
    retry_count     = state.get("retry_count", 0)
    used_web_search = state.get("used_web_search", False)

    if relevant_docs:
        print(f"[Router] Relevant docs found → generation")
        return "generation"

    if used_web_search:
        print(f"[Router] Web search also failed → END")
        return "end_no_answer"

    if retry_count < MAX_RETRIES:
        print(f"[Router] No relevant docs — retry {retry_count}/{MAX_RETRIES} → query_analysis")
        return "query_analysis"

    print(f"[Router] Max retries reached → web_search")
    return "web_search"


def route_after_hallucination_check(state: RAGState) -> str:
    """
    Routes after hallucination check.
    Always ends — either clean answer or flagged answer with warning.
    hallucination_flag is handled in the API response layer.
    """
    hallucination_flag = state.get("hallucination_flag", False)
    if hallucination_flag:
        print(f"[Router] ⚠ Hallucination detected — returning answer with warning")
    else:
        print(f"[Router] ✓ Answer grounded — returning clean answer")
    return "end"

def route_after_query_analysis(state: RAGState) -> str:
    query_type = state.get("query_type", "")
    if query_type == "chitchat":
        print(f"[Router] Chitchat detected → generation directly")
        return "generation"
    print(f"[Router] Technical query → retrieval")
    return "retrieval"

# ── No Answer Node ─────────────────────────────────────────────────────────────

def no_answer_node(state: RAGState) -> dict:
    """
    Called when both Chroma and web search fail to find relevant docs.
    Returns a clear 'I don't know' response.
    """
    print(f"\n[NoAnswer] No relevant docs found from Chroma or web search")
    return {
        "answer"          : "I don't have enough information to answer this question. Please try rephrasing or ask about a different topic.",
        "sources"         : [],
        "hallucination_flag": False,
    }


# ── Reset Node ─────────────────────────────────────────────────────────────────

def reset_state_node(state: RAGState) -> dict:
    """
    Resets per-query state fields at the start of each new question.
    Ensures used_web_search, retry_count, etc. don't carry over between turns.
    """
    return {
        "used_web_search"      : False,
        "web_results"          : [],
        "documents"            : [],
        "relevant_docs"        : [],
        "retry_count"          : 0,
        "grade_reasoning"      : [],
        "hallucination_flag"   : False,
        "hallucination_attempts": 0,
        "sources"              : [],
        "answer"               : "",
    }


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    # ── Add Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("reset",               reset_state_node)
    graph.add_node("query_analysis",      query_analysis_node)
    graph.add_node("retrieval",           retrieval_node)
    graph.add_node("grading",             grading_node)
    graph.add_node("generation",          generation_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("web_search",          web_search_node)
    graph.add_node("end_no_answer",       no_answer_node)

    # ── Entry Point ────────────────────────────────────────────────────────────
    graph.set_entry_point("reset")

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_edge("reset",          "query_analysis")
    
    graph.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "generation": "generation",
            "retrieval" : "retrieval",
        }
    )
    graph.add_edge("retrieval",      "grading")
    graph.add_edge("web_search",     "grading")
    graph.add_edge("generation",     "hallucination_check")
    graph.add_edge("end_no_answer",  END)

    # ── Conditional Edges ──────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generation"   : "generation",
            "query_analysis": "query_analysis",
            "web_search"   : "web_search",
            "end_no_answer": "end_no_answer",
        }
    )

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "end": END,
        }
    )

    return graph.compile()


rag_graph = build_graph()