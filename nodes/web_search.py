"""
nodes/web_search.py — Web Search Fallback
Triggered when retry_count >= 2 and Chroma has no relevant docs.
Searches the web using Serper API and returns results as Documents.
Input  state keys: rewritten_query
Output state keys: web_results, used_web_search
"""
import os
import sys
import requests
from pathlib import Path
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parent.parent))
from state import RAGState

SERPER_API_URL = "https://google.serper.dev/search"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MAX_RESULTS    = 5


def search_web(query: str) -> list[Document]:
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": MAX_RESULTS,
    }
    response = requests.post(SERPER_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    docs = []
    for result in data.get("organic", [])[:MAX_RESULTS]:
        title   = result.get("title", "")
        snippet = result.get("snippet", "")
        link    = result.get("link", "")

        doc = Document(
            page_content=f"{title}\n\n{snippet}",
            metadata={
                "source"     : link,
                "source_type": "web_search",
                "title"      : title,
            }
        )
        docs.append(doc)
    return docs


def web_search_node(state: RAGState) -> dict:
    rewritten_query = state["rewritten_query"]

    print(f"\n[WebSearch] Chroma exhausted — falling back to Serper web search...")
    print(f"[WebSearch] Query: '{rewritten_query}'")

    try:
        web_results = search_web(rewritten_query)
        print(f"[WebSearch] Found {len(web_results)} result(s)")
        for doc in web_results:
            print(f"  - {doc.metadata.get('source', 'unknown')}")

    except Exception as e:
        print(f"[WebSearch] ✗ Search failed: {e}")
        web_results = []

    return {
        "web_results"    : web_results,
        "used_web_search": True,  # always True — we attempted web search
        "documents"      : web_results,
    }