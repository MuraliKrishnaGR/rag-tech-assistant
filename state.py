"""
state.py — LangGraph State Schema
Defines all data that flows between nodes in the RAG workflow.
See README.md for full architecture and flow diagram.
"""

from typing_extensions import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str
    session_id: str
    user_id: str
    chat_history: list[dict]

    # ── Query Analysis ─────────────────────────────────────────────────────────
    rewritten_query: str
    query_type: str

    # ── Retrieval ──────────────────────────────────────────────────────────────
    documents: list[Document]

    # ── Grading ────────────────────────────────────────────────────────────────
    relevant_docs: list[Document]
    retry_count: int
    grade_reasoning: list[dict]
    # Format: [{"source": "...", "grade": "YES", "reason": "..."}]

    # ── Web Search Fallback ────────────────────────────────────────────────────
    web_results: list[Document]
    used_web_search: bool

    # ── Generation ─────────────────────────────────────────────────────────────
    answer: str
    sources: list[str]

    # ── Hallucination Check ────────────────────────────────────────────────────
    hallucination_flag: bool
    hallucination_attempts: int
    # Tracks regeneration attempts after hallucination detected
    # Max 1 attempt → regenerate with stricter prompt
    # Still hallucinated after retry → return answer with ⚠ warning