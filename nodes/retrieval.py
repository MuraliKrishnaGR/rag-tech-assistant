"""
nodes/retrieval.py — Node 2: Retrieval

Input state:  rewritten_query
Output state: documents
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import retriever
from state import RAGState


def retrieval_node(state: RAGState) -> dict:
    rewritten_query = state["rewritten_query"]

    print(f"[Retrieval] Searching for: '{rewritten_query}'")

    documents = retriever.invoke(rewritten_query)

    print(f"[Retrieval] Found {len(documents)} chunks")
    for doc in documents:
        print(f"  - {doc.metadata.get('source', 'unknown')}")

    return {
        "documents": documents,
    }