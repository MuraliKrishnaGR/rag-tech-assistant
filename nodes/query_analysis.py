"""
nodes/query_analysis.py — Node 1: Query Analysis

Input state:  question, chat_history, retry_count, grade_reasoning
Output state: rewritten_query, query_type

On first run (retry_count = 0):
  - Rewrites query using chat history context
  - Classifies query type:
      conceptual | how-to | troubleshooting | api-reference | chitchat
  - chitchat → routed directly to generation (skips retrieval)

On retry (retry_count > 0):
  - Uses grade_reasoning from previous attempt to write a smarter query
  - YES reasons → find more content like this
  - NO reasons  → avoid content like this
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import llm
from state import RAGState


# ── First attempt prompt ───────────────────────────────────────────────────────
PROMPT_FIRST = """\
You are a query analysis assistant for a technical documentation Q&A system.

Chat history (last few exchanges):
{chat_history}

User's current question: {question}

Your tasks:
1. Rewrite the question to be self-contained and specific.
   - Resolve any pronouns or references using chat history
   - Add synonyms or related terms to improve retrieval
   - If chitchat, keep the rewritten query same as original

2. Classify the query type as ONE of:
   - conceptual      → user wants to understand a concept
   - how-to          → user wants step-by-step instructions
   - troubleshooting → user has an error or something not working
   - api-reference   → user wants specific API details, parameters, return types
   - chitchat        → greeting, small talk, or clarification not requiring docs

Respond in this exact format and nothing else:
REWRITTEN: <rewritten query>
TYPE: <query type>"""


# ── Retry prompt ───────────────────────────────────────────────────────────────
PROMPT_RETRY = """\
You are a query analysis assistant for a technical documentation Q&A system.

Chat history (last few exchanges):
{chat_history}

User's current question: {question}

Previous retrieval attempt failed to find relevant documents.
Here is the grading feedback from the previous attempt:

{grade_reasoning}

Use this feedback to rewrite a BETTER query:
- Look at YES reasons → find more content similar to those chunks
- Look at NO reasons  → avoid retrieving content like those chunks
- Be more specific, use different terminology if needed

Also classify the query type as ONE of:
   - conceptual      → user wants to understand a concept
   - how-to          → user wants step-by-step instructions
   - troubleshooting → user has an error or something not working
   - api-reference   → user wants specific API details, parameters, return types
   

Respond in this exact format and nothing else:
REWRITTEN: <rewritten query>
TYPE: <query type>"""


def format_chat_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "No previous conversation."
    lines = []
    for msg in chat_history[-6:]:  # last 3 exchanges
        role = "User" if msg["role"] == "human" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def format_grade_reasoning(grade_reasoning: list[dict]) -> str:
    """
    Format grade_reasoning into a readable string for the retry prompt.
    Example output:
      ✓ YES - langchain/overview → Explains StateGraph nodes clearly
      ✗ NO  - langchain/quickstart → About installation, not relevant
    """
    if not grade_reasoning:
        return "No grading feedback available."
    lines = []
    for item in grade_reasoning:
        icon  = "✓" if item["grade"] == "YES" else "✗"
        lines.append(f"  {icon} {item['grade']} - {item['source']} → {item['reason']}")
    return "\n".join(lines)


def query_analysis_node(state: RAGState) -> dict:
    """
    Query Analysis Node.

    First run  → rewrites query using chat history
    On retry   → rewrites query using grade_reasoning feedback
                 so each retry is smarter than the last
    """
    question        = state["question"]
    chat_history    = format_chat_history(state.get("chat_history", []))
    retry_count     = state.get("retry_count", 0)
    grade_reasoning = state.get("grade_reasoning", [])

    print(f"\n[QueryAnalysis] retry_count={retry_count}")

    # Choose prompt based on whether this is a retry
    if retry_count > 0 and grade_reasoning:
        print(f"[QueryAnalysis] Using grade_reasoning to improve query...")
        prompt = PROMPT_RETRY.format(
            question=question,
            chat_history=chat_history,
            grade_reasoning=format_grade_reasoning(grade_reasoning),
        )
    else:
        prompt = PROMPT_FIRST.format(
            question=question,
            chat_history=chat_history,
        )

    response = llm.invoke(prompt).content.strip()

    # Parse response
    rewritten_query = question
    query_type      = "conceptual"

    for line in response.splitlines():
        if line.startswith("REWRITTEN:"):
            rewritten_query = line.replace("REWRITTEN:", "").strip()
        elif line.startswith("TYPE:"):
            query_type = line.replace("TYPE:", "").strip().lower()

    # Validate query type
    valid_types = {"conceptual", "how-to", "troubleshooting", "api-reference", "chitchat"}
    if query_type not in valid_types:
        query_type = "conceptual"

    print(f"[QueryAnalysis] type='{query_type}'")
    print(f"[QueryAnalysis] rewritten='{rewritten_query}'")

    return {
        "rewritten_query": rewritten_query,
        "query_type": query_type,
    }