"""
nodes/hallucination.py — Hallucination Check

Verifies the generated answer is grounded in the retrieved context.
Inspired by Self-RAG.

If hallucination detected:
  - Attempt 1 → regenerate with stricter prompt
  - Attempt 2 → still hallucinated → return answer with ⚠ warning

Input  state keys: question, answer, relevant_docs, web_results,
                   used_web_search, hallucination_attempts
Output state keys: answer, hallucination_flag, hallucination_attempts
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import llm
from state import RAGState


CHECK_PROMPT = """\
You are a hallucination detector for a RAG system.

Context:
{context}

Generated Answer:
{answer}

Instructions:
- Check if the answer is based on the provided context
- If the answer is grounded in the context, even if synthesized or 
  elaborated into code examples → NOT hallucinated
- If the answer introduces information completely absent from 
  the context → hallucinated

Respond in this exact format:
HALLUCINATED: YES or NO
REASON: <one sentence explaining your decision>"""


REGENERATE_PROMPT = """\
You are a technical documentation assistant.

Your previous answer contained claims not fully supported by the context.
Regenerate the answer using ONLY information explicitly stated in the context below.
Do not add anything that is not present in the context.

Context:
{context}

Original question: {question}

Previous answer (contained hallucinations):
{answer}

Regenerated answer (strictly grounded):"""


def build_context(docs: list) -> str:
    """Combine all doc content into a single context string with sources."""
    return "\n\n---\n\n".join([
        f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])


def check_hallucination(answer: str, context: str) -> tuple[bool, str]:
    """
    Check if the answer is grounded in the context.
    Returns (hallucinated: bool, reason: str)
    """
    prompt = CHECK_PROMPT.format(context=context, answer=answer)
    response = llm.invoke(prompt).content.strip()

    hallucinated = False
    reason = ""

    for line in response.splitlines():
        if line.startswith("HALLUCINATED:"):
            hallucinated = "YES" in line.upper()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return hallucinated, reason


def hallucination_check_node(state: RAGState) -> dict:
    """
    Hallucination Check Node.

    Step 1 — Check if answer is grounded in context
    Step 2 — If hallucinated and first attempt → regenerate with stricter prompt
    Step 3 — If still hallucinated after retry → return answer with ⚠ warning
    """
    answer                 = state.get("answer", "")
    question               = state.get("question", "")
    used_web_search        = state.get("used_web_search", False)
    hallucination_attempts = state.get("hallucination_attempts", 0)

    docs = state.get("web_results", []) if used_web_search else state.get("relevant_docs", [])

    print(f"\n[HallucinationCheck] Checking answer (attempt {hallucination_attempts + 1})...")

    if not docs or not answer:
        print("[HallucinationCheck] No context or answer — skipping")
        return {"hallucination_flag": False, "hallucination_attempts": hallucination_attempts}

    context = build_context(docs)

    # Step 1: Check hallucination
    hallucinated, reason = check_hallucination(answer, context)

    print(f"[HallucinationCheck] hallucinated={hallucinated}")
    print(f"[HallucinationCheck] reason={reason}")

    # Step 2: If hallucinated and first attempt → regenerate
    if hallucinated and hallucination_attempts == 0:
        print(f"[HallucinationCheck] Regenerating with stricter prompt...")

        regen_prompt = REGENERATE_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )
        new_answer = llm.invoke(regen_prompt).content.strip()

        # Check regenerated answer
        still_hallucinated, reason = check_hallucination(new_answer, context)

        print(f"[HallucinationCheck] After regeneration: hallucinated={still_hallucinated}")
        print(f"[HallucinationCheck] reason={reason}")

        return {
            "answer": new_answer,
            "hallucination_flag": still_hallucinated,
            "hallucination_attempts": hallucination_attempts + 1,
        }

    # Step 3: Still hallucinated after retry → flag it, warn user in API response
    if hallucinated:
        print("[HallucinationCheck] ⚠ Still hallucinated after retry — flagging answer")

    return {
        "hallucination_flag": hallucinated,
        "hallucination_attempts": hallucination_attempts + 1,
    }