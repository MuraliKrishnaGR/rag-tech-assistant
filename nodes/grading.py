"""
nodes/grading.py — Node 3: Document Grading

Self-corrective component of the RAG pipeline.
Grades each retrieved chunk individually as relevant or irrelevant.

Input  state keys: question, rewritten_query, documents, retry_count
Output state keys: relevant_docs, retry_count, grade_reasoning

Routing (in graph.py):
  relevant_docs found     → Generation
  no relevant_docs found  → retry_count < 2 → Retrieval (rewrite + retry)
  no relevant_docs found  → retry_count >= 2 → Web Search fallback
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import llm
from state import RAGState


PROMPT = """\
You are a document relevance grader for a technical Q&A system.

Original question : {question}
Rewritten query   : {rewritten_query}

Retrieved document chunk:
{content}

Is this chunk relevant to answering the original question?
A chunk is relevant if it directly helps answer the question — even partially.
A chunk is irrelevant if it is off-topic, too general, or unrelated.

Respond in this exact format:
GRADE: YES or NO
REASON: <one sentence explaining why>"""


def grading_node(state: RAGState) -> dict:
    """
    Document Grading Node.

    Grades each retrieved chunk individually using the LLM.
    Keeps grade reasoning for each chunk — useful for evaluators and debugging.
    Increments retry_count only when NO relevant docs found.
    """
    question        = state["question"]
    rewritten_query = state["rewritten_query"]
    documents       = state.get("documents", [])
    retry_count     = state.get("retry_count", 0)

    print(f"\n[Grading] Grading {len(documents)} chunk(s) individually...")

    relevant_docs   = []
    grade_reasoning = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")

        prompt = PROMPT.format(
            question=question,
            rewritten_query=rewritten_query,
            content=doc.page_content,
        )

        response = llm.invoke(prompt).content.strip()

        # Parse GRADE and REASON from response
        grade  = "YES"    # safe default
        reason = "Could not parse reason"

        for line in response.splitlines():
            if line.startswith("GRADE:"):
                raw = line.replace("GRADE:", "").strip().upper()
                grade = "YES" if raw.startswith("YES") else "NO"
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        is_relevant = grade == "YES"

        if is_relevant:
            relevant_docs.append(doc)

        grade_reasoning.append({
            "source": source,
            "grade": grade,
            "reason": reason,
        })

        print(f"  {'✓' if is_relevant else '✗'} [{grade}] {source}")
        print(f"      → {reason}")

    print(f"\n[Grading] {len(relevant_docs)}/{len(documents)} chunk(s) relevant")

    # Increment retry_count only when ALL chunks are irrelevant
    if not relevant_docs:
        retry_count += 1
        print(f"[Grading] No relevant docs — retry_count now {retry_count}")

    return {
        "relevant_docs": relevant_docs,
        "retry_count": retry_count,
        "grade_reasoning": grade_reasoning,
    }