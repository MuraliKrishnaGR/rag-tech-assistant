"""
nodes/generation.py — Node 4: Generation

Generates the final answer grounded in relevant retrieved documents.
Includes inline citations referencing source documents.

Input  state keys: question, query_type, relevant_docs,
                   web_results, used_web_search, chat_history
Output state keys: answer, sources
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import llm
from state import RAGState


PROMPT = """\
You are a technical documentation assistant. Answer the user's question \
based ONLY on the provided context below.

Question type: {query_type}
- conceptual      → explain clearly with examples
- how-to          → give step-by-step instructions
- troubleshooting → diagnose the problem and provide fixes
- api-reference   → be precise and technical, include parameters and return types

Chat history (for context):
{chat_history}

User question: {question}

Context:
{context}

Instructions:
- Answer based ONLY on the context above
- Include inline citations like [source: <filename or url>]
- If context is insufficient to fully answer, say so explicitly
- Do NOT make up information not present in the context"""


CHITCHAT_PROMPT = """\
You are a helpful technical documentation assistant.

Chat history:
{chat_history}

User said: {question}

Respond naturally and conversationally.
- If it's a greeting, greet back warmly
- If the user asks what you can help with, explain you can answer 
  questions about any technical documentation that has been ingested
- Keep it brief and friendly"""


def format_chat_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "No previous conversation."
    lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "human" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def format_context(docs: list) -> tuple[str, list[str]]:
    """Format docs into context string and extract unique sources."""
    context_parts = []
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[source: {source}]\n{doc.page_content}")
        if source not in sources:
            sources.append(source)
    return "\n\n---\n\n".join(context_parts), sources


def generation_node(state: RAGState) -> dict:
    question        = state["question"]
    query_type      = state.get("query_type", "conceptual")
    chat_history    = format_chat_history(state.get("chat_history", []))
    used_web_search = state.get("used_web_search", False)

    # ── Chitchat — respond directly, no context needed ────────────────────────
    if query_type == "chitchat":
        print(f"\n[Generation] Chitchat detected — responding directly")
        prompt = CHITCHAT_PROMPT.format(
            chat_history=chat_history,
            question=question,
        )
        answer = llm.invoke(prompt).content.strip()
        print(f"[Generation] Answer generated ({len(answer)} chars)")
        return {
            "answer" : answer,
            "sources": [],
        }

    # ── Technical query — use retrieved context ───────────────────────────────
    docs = state.get("web_results", []) if used_web_search else state.get("relevant_docs", [])

    print(f"\n[Generation] Generating answer using {len(docs)} chunk(s)...")
    print(f"  source(s): {[doc.metadata.get('source', 'unknown') for doc in docs]}")
    print(f"  web search used: {used_web_search}")

    # Guard — if no context, return honest response
    if not docs:
        print("[Generation] No context available — returning fallback response")
        return {
            "answer" : "I don't have enough information to answer this question.",
            "sources": [],
        }

    context, sources = format_context(docs)

    prompt = PROMPT.format(
        query_type=query_type,
        chat_history=chat_history,
        question=question,
        context=context,
    )

    answer = llm.invoke(prompt).content.strip()

    print(f"[Generation] Answer generated ({len(answer)} chars)")
    print(f"[Generation] Sources: {sources}")

    return {
        "answer" : answer,
        "sources": sources,
    }