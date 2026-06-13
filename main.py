"""
main.py — FastAPI Application Entry Point

Endpoints:
  POST   /query                    → Submit a question, get answer with sources
  POST   /ingest/urls              → Ingest documents from URLs (JSON list)
  POST   /ingest/file              → Ingest a file upload
  GET    /documents                → List all indexed documents
  POST   /feedback                 → Submit feedback on an answer
  GET    /sessions/{session_id}    → Get conversation history for a session
  GET    /sessions/user/{user_id}  → Get all sessions for a user
  DELETE /sessions/{session_id}    → Delete a conversation session
  GET    /health                   → Health check
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from graph import rag_graph
from db import (
    init_db,
    list_documents,
    save_message,
    get_session_messages,
    get_user_sessions,
    delete_session,
    save_feedback,
)
from scripts.indexer import ingest_urls, ingest_file, ingest_default_urls


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at startup:
    1. Initialize PostgreSQL tables
    2. Ingest default LangChain/LangGraph docs into Chroma
    """
    print("\n[Startup] Initializing database...")
    init_db()

    print("[Startup] Ingesting default docs...")
    ingest_default_urls()

    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="A self-corrective RAG system built with LangGraph and FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    session_id: str
    user_id:    str

class QueryResponse(BaseModel):
    answer:             str
    sources:            list[str]
    used_web_search:    bool
    hallucination_flag: bool
    grade_reasoning:    list[dict]
    session_id:         str

class IngestURLRequest(BaseModel):
    urls: list[str]

class FeedbackRequest(BaseModel):
    question: str
    answer:   str
    rating:   str           # "up" or "down"
    comment:  Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a question and get an answer with sources.
    Runs the full LangGraph RAG workflow.
    Saves question + answer to PostgreSQL for conversation memory.
    """
    # Fetch chat history from PostgreSQL
    chat_history = get_session_messages(request.session_id)

    # Run LangGraph workflow
    result = rag_graph.invoke({
        "question"    : request.question,
        "session_id"  : request.session_id,
        "user_id"     : request.user_id,
        "chat_history": chat_history,
    })

    answer             = result.get("answer", "I don't have enough information to answer this question.")
    hallucination_flag = result.get("hallucination_flag", False)

    # Append warning to answer if hallucination detected
    if hallucination_flag:
        answer += "\n\n⚠ Warning: This answer may not be fully grounded in the retrieved documents."

    # Save question + answer to PostgreSQL
    save_message(request.session_id, request.user_id, "human", request.question)
    save_message(request.session_id, request.user_id, "ai", answer)

    return QueryResponse(
        answer             = answer,
        sources            = result.get("sources", []),
        used_web_search    = result.get("used_web_search", False),
        hallucination_flag = hallucination_flag,
        grade_reasoning    = result.get("grade_reasoning", []),
        session_id         = request.session_id,
    )


@app.post("/ingest/urls")
async def ingest_urls_endpoint(request: IngestURLRequest):
    """
    Ingest documents from a list of URLs.
    Accepts: {"urls": ["https://...", "https://..."]}
    """
    if not request.urls:
        raise HTTPException(status_code=400, detail="Provide at least one URL.")

    results = ingest_urls(request.urls)
    return {"status": "success", "results": results}


@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    """
    Ingest a file upload into the vector store.
    Supported: .pdf, .docx, .md, .txt, .html
    """
    file_bytes = await file.read()
    result = ingest_file(file_bytes, file.filename)
    return {"status": "success", "result": result}


@app.get("/documents")
def documents():
    """List all indexed documents in the vector store."""
    docs = list_documents()
    return {
        "count"    : len(docs),
        "documents": [dict(doc) for doc in docs],
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Submit thumbs up/down feedback on an answer."""
    if request.rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'.")

    save_feedback(
        question=request.question,
        answer  =request.answer,
        rating  =request.rating,
        comment =request.comment,
    )
    return {"status": "success", "message": "Feedback saved."}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get full conversation history for a session."""
    history = get_session_messages(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "history": history}


@app.get("/sessions/user/{user_id}")
def get_user_session_list(user_id: str):
    """Get all sessions for a user (browser cookie)."""
    sessions = get_user_sessions(user_id)
    return {
        "user_id" : user_id,
        "count"   : len(sessions),
        "sessions": [dict(s) for s in sessions],
    }


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """Delete a conversation session and its history."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "success", "message": f"Session '{session_id}' deleted."}


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)