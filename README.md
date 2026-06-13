# rag-tech-assistant

A technical documentation Q&A assistant that retrieves relevant chunks, grades them for relevance, and generates grounded answers with citations.

Built with **LangGraph**, **FastAPI**, and **Streamlit**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [State Schema Design](#state-schema-design)
- [LangGraph Workflow](#langgraph-workflow)
- [Document Ingestion Pipeline](#document-ingestion-pipeline)
- [API Endpoints](#api-endpoints)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Example API Requests and Responses](#example-api-requests-and-responses)
- [Thought Process and Architecture Reasoning](#thought-process-and-architecture-reasoning)
- [Document Corpus](#document-corpus)
- [Assumptions](#assumptions)
- [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
- [Chunking and Embedding Strategy](#chunking-and-embedding-strategy)
- [What I Would Improve With More Time](#what-i-would-improve-with-more-time)

---

## Overview

The system accepts a natural language question, retrieves the most relevant chunks from a vector store, grades each chunk for relevance using an LLM, generates a grounded answer with inline citations, and verifies the answer for hallucinations. If retrieval fails, the system rewrites the query and retries. If all retries fail, it falls back to a web search. If web search also fails, it returns an honest "I don't know" response.

---

## Architecture

![Architecture Diagram](architecture.png)

> Full architecture diagram. See `architecture.drawio` to edit.

---

## State Schema Design

The state schema is the most critical design decision in a LangGraph pipeline. Every field was chosen deliberately based on what data needs to flow between nodes.

```python
class RAGState(TypedDict):
    # Input — set once by main.py before graph starts
    question: str           # original user question, never mutated
    session_id: str         # identifies the conversation
    user_id: str            # identifies the user (for DB storage)
    chat_history: list[dict]  # history loaded from PostgreSQL, LLM prompt uses last 6 (3 exchanges)

    # Query Analysis — produced by Node 1
    rewritten_query: str    # expanded query for better retrieval
    query_type: str         # conceptual | how-to | troubleshooting | api-reference | chitchat

    # Retrieval — produced by Node 2
    documents: list[Document]  # raw top-k chunks from Chroma

    # Grading — produced by Node 3
    relevant_docs: list[Document]  # subset graded as relevant
    retry_count: int               # tracks rewrite+retry cycles
    grade_reasoning: list[dict]    # per-chunk grade + reason (for retry rewriting + UI)

    # Web Search Fallback
    web_results: list[Document]  # Serper results as Documents
    used_web_search: bool         # tells generation which docs to use

    # Generation — produced by Node 4
    answer: str
    sources: list[str]

    # Hallucination Check
    hallucination_flag: bool
    hallucination_attempts: int
```

### How retry_count works

`retry_count` starts at 0. When `grading_node` finds no relevant documents, it increments `retry_count` and returns. The conditional edge in `graph.py` reads `retry_count`:

- `retry_count < 2` → route back to `query_analysis` for a smarter rewrite
- `retry_count >= 2` → route to `web_search` fallback

This means the graph can self-correct up to 2 times before escalating to web search. Each retry is smarter than the last because `query_analysis` uses `grade_reasoning` from the failed attempt to understand what kind of content to look for (or avoid).

---

## LangGraph Workflow

### Node 1: Query Analysis (`nodes/query_analysis.py`)

- Classifies query type: `conceptual`, `how-to`, `troubleshooting`, `api-reference`, `chitchat`
- Rewrites/expands the query to improve retrieval recall
- On retry: uses `grade_reasoning` from the previous failed attempt to write a smarter query
- Chitchat queries skip retrieval entirely and go directly to generation

### Node 2: Retrieval (`nodes/retrieval.py`)

- Uses LangChain's `retriever.invoke()` with Chroma Cloud
- Returns top-5 chunks with source metadata
- Uses the `rewritten_query` (not the original) for better recall

### Node 3: Document Grading (`nodes/grading.py`) — Self-Corrective Component

- Grades each chunk **individually** using a separate LLM call
- Returns `GRADE: YES/NO` and `REASON` for each chunk
- Reasons are stored in `grade_reasoning` — used by query_analysis on retry and displayed in the UI
- Increments `retry_count` only when ALL chunks are irrelevant
- Conditional edge routes based on `relevant_docs` and `retry_count`

### Node 4: Generation (`nodes/generation.py`)

- Uses `relevant_docs` or `web_results` depending on `used_web_search`
- Adjusts answer style based on `query_type` (step-by-step for how-to, precise for api-reference, etc.)
- Uses `chat_history` for follow-up question context
- Includes inline citations `[source: <url or filename>]`
- Handles chitchat with a conversational prompt (no context needed)

### Node 5: Hallucination Check (`nodes/hallucination.py`)

- Checks if the generated answer is grounded in the retrieved context
- If hallucinated on first attempt → regenerates with a stricter prompt
- If still hallucinated after regeneration → returns answer with `⚠ warning`
- Inspired by Self-RAG

### Node 6: Web Search (`nodes/web_search.py`)

- Triggered when `retry_count >= 2` and Chroma has no relevant docs
- Uses Serper API to search the web
- Converts results to LangChain `Document` objects — generation handles them identically to Chroma chunks
- Web results also go through grading before generation

---

## Document Ingestion Pipeline

### How it works

1. Load documents from URLs (via `WebBaseLoader`) or file uploads (`.pdf`, `.docx`, `.md`, `.txt`, `.html`)
2. Extract text and compute a document-level checksum
3. Compare checksum against PostgreSQL — skip if unchanged (free, no embedding cost)
4. Split into chunks using `RecursiveCharacterTextSplitter`
5. Compute deterministic chunk IDs: `md5(source + chunk_content)`
6. Upsert into Chroma Cloud with chunk IDs — only new/changed chunks get embedded
7. Register or update source in PostgreSQL

### Duplicate and Update Handling

The pipeline uses two layers:

- **Layer 1 (document level)**: SHA-256 checksum of normalized extracted text → stored in PostgreSQL. If unchanged, skip entirely (zero embedding cost).
- **Layer 2 (chunk level)**: Deterministic chunk IDs in Chroma. On re-ingestion of updated content, only new chunks get embedded, stale chunks get deleted. This minimizes embedding API costs even when content changes.

### Default Corpus

The system pre-loads these URLs on startup:

- `https://docs.langchain.com/oss/python/langchain/overview`
- `https://docs.langchain.com/oss/python/langchain/quickstart`
- `https://docs.langchain.com/oss/python/langchain/agents`
- `https://docs.langchain.com/oss/python/langgraph/overview`
- `https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph`

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/query` | Submit a question, returns answer with sources |
| POST | `/ingest/urls` | Ingest from a list of URLs |
| POST | `/ingest/file` | Upload a file (.pdf, .docx, .md, .txt, .html) |
| GET | `/documents` | List all indexed documents and chunk counts |
| POST | `/feedback` | Submit thumbs up/down feedback on an answer |
| GET | `/sessions/{session_id}` | Get conversation history for a session |
| GET | `/sessions/user/{user_id}` | Get all sessions for a user |
| DELETE | `/sessions/{session_id}` | Delete a conversation session |
| GET | `/health` | Health check |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- PostgreSQL (local) or Neon (cloud)
- API keys: Groq, Cohere, Chroma Cloud, Serper

### Environment Variables

Create a `.env` file:

```env
# LLM
GROQ_API_KEY=your_groq_api_key

# Embeddings
COHERE_API_KEY=your_cohere_api_key

# Chroma Cloud
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database

# PostgreSQL (local)
DB_HOST=localhost
DB_NAME=rag_assistant
DB_USER=postgres
DB_PASSWORD=password
DB_PORT=5432

# OR Neon (production)
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/dbname?sslmode=require

# Web Search (optional — for fallback)
SERPER_API_KEY=your_serper_api_key
```

### Install Dependencies

```bash
python -m venv tenv
source tenv/bin/activate  # Windows: tenv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Start FastAPI

```bash
uvicorn main:app --reload --port 8000
```

This will:
- Initialize PostgreSQL tables
- Ingest default LangChain/LangGraph docs into Chroma Cloud
- Start the API server at `http://localhost:8000`

### 2. Start Streamlit UI

In a separate terminal:

```bash
streamlit run app.py
```

### 3. API Docs

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Example API Requests and Responses

### POST /query

**Request:**
```json
{
  "question": "How do I create a StateGraph in LangGraph?",
  "session_id": "abc-123",
  "user_id": "user-456"
}
```

**Response:**
```json
{
  "answer": "To create a StateGraph in LangGraph, follow these steps:\n\n1. Define your state schema using TypedDict...\n[source: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph]",
  "sources": ["https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph"],
  "used_web_search": false,
  "hallucination_flag": false,
  "grade_reasoning": [
    {
      "source": "https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph",
      "grade": "YES",
      "reason": "The chunk explains how to instantiate a StateGraph directly addressing the question."
    }
  ],
  "session_id": "abc-123"
}
```

### POST /ingest/urls

**Request:**
```json
{
  "urls": ["https://huggingface.co/docs/transformers/training"]
}
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "status": "ingested",
      "source": "https://huggingface.co/docs/transformers/training",
      "chunks_added": 17,
      "chunks_deleted": 0,
      "chunks_total": 17
    }
  ]
}
```

### GET /documents

**Response:**
```json
{
  "count": 5,
  "documents": [
    {
      "source_name": "https://docs.langchain.com/oss/python/langgraph/overview",
      "source_type": "url",
      "chunk_count": 12,
      "ingested_at": "2026-06-13T10:00:00"
    }
  ]
}
```

### POST /ingest/file

Upload a file via multipart form:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@deep_learning.md"
```

**Response:**
```json
{
  "status": "success",
  "result": {
    "status": "ingested",
    "source": "deep_learning.md",
    "chunks_added": 23,
    "chunks_deleted": 0,
    "chunks_total": 23
  }
}
```

### POST /feedback

**Request:**
```json
{
  "question": "How do I create a StateGraph?",
  "answer": "To create a StateGraph...",
  "rating": "up",
  "comment": "Very helpful!"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback saved."
}
```

---

## Thought Process and Architecture Reasoning

The assignment asks for a self-corrective RAG pipeline. The key question is: what does "self-corrective" actually mean in practice?

Most RAG implementations retrieve chunks and generate regardless of whether the chunks are relevant. The self-correction in this system happens at three levels:

- **Query level** — the query is rewritten and expanded before retrieval to improve recall
- **Grading level** — each retrieved chunk is individually evaluated by an LLM before generation
- **Retry level** — when grading fails, the system rewrites the query using the grading feedback and retries, rather than blindly rewriting or giving up

This three-level correction is what separates this from a simple retrieve → grade → generate chain.

**Why store grade_reasoning in state?**
When grading fails, most systems blindly rewrite the query. Instead, I store each chunk's grade and reason in state. On retry, `query_analysis` reads these reasons — YES reasons tell it what content to find more of, NO reasons tell it what to avoid. Each retry is smarter than the last because it learns from its own failure.

**Why is retry_count in state and not a local variable?**
The retry loop spans multiple node executions. A local variable would reset every time a node runs. Storing `retry_count` in state means the graph can track how many times grading has failed across the entire request cycle and route accordingly.

**Why keep PostgreSQL outside the graph?**
Chat history is loaded once before the graph starts and saved once after it finishes. The graph itself never touches the database — it's a pure function of its input state. This keeps the pipeline clean, testable, and easy to reason about.

**Why two layers of duplicate detection in ingestion?**
A document-level checksum catches unchanged re-ingestions at zero embedding cost. Chunk-level checksums handle updates — only changed chunks get re-embedded, stale ones get deleted. This makes ingestion safe to call repeatedly without inflating the vector store.

**Why web search as a last resort?**
The vector store contains curated documentation. Web search is uncontrolled and noisy. It only triggers after two failed retrieval attempts — the system exhausts its own knowledge before going to the open web.

---

## Document Corpus

The system pre-loads 5 LangChain and LangGraph documentation pages on startup via `scripts/indexer.py`. These serve as the default corpus:

- LangChain Overview
- LangChain Quickstart
- LangChain Agents
- LangGraph Overview
- Thinking in LangGraph

Additional documents can be ingested at any time via `POST /ingest/urls` or `POST /ingest/file`. The corpus is not fixed — it grows as users add documents.

---

## Design Decisions and Tradeoffs

### Why grade each chunk individually?

Grading each chunk with a separate LLM call is more accurate than batch grading. The LLM can focus on a single chunk without being confused by other chunks. The tradeoff is cost — 5 chunks = 5 LLM calls. For this use case, accuracy matters more than cost.

### Why store grade_reasoning in state?

`grade_reasoning` serves three purposes:
1. **Smarter retries** — on retry, `query_analysis` reads the YES/NO reasons from the previous grading attempt. Instead of blindly rewriting the query, it knows what kind of content to look for and what to avoid. Each retry is genuinely smarter than the last.
2. **Query analysis context** — even on the first run, grade_reasoning from a previous conversation turn can inform how the query is analysed in follow-up questions.
3. **Transparency** — the Streamlit UI exposes grade_reasoning so users can see exactly why each chunk was kept or dropped, making the self-corrective behaviour observable and debuggable.

### Why limit chat history to 6 messages in the LLM prompt?

The full conversation history is loaded from PostgreSQL and stored in state. When building the LLM prompt, `format_chat_history()` slices the last 6 messages (3 exchanges). This is a deliberate tradeoff — passing the full history would bloat the prompt and increase token cost on every query. 3 exchanges is enough to resolve pronouns and follow-up references in most conversations.

### Why PostgreSQL for chat history instead of LangGraph's add_messages?

`add_messages` with `Annotated[Sequence[BaseMessage], add_messages]` is designed for agent flows where the LangGraph graph itself is the conversation loop. In this pipeline, the graph runs once per question — it's not a loop. Chat history is loaded once from PostgreSQL before the graph starts, passed as a read-only field, and saved back to PostgreSQL after the graph finishes. This keeps DB interaction outside the graph entirely — clean separation of concerns.

### Why Chroma Cloud?

Chroma Cloud was chosen for persistent vector storage without managing any infrastructure. The `langchain_chroma` integration works out of the box — the same `vectorstore.similarity_search()` and `vectorstore.add_documents()` calls work identically whether you're running locally or in production. No custom serialization, no self-hosted server, no ops overhead. The free tier is sufficient for this project's corpus size.

### Why Cohere embeddings?

Cohere's `embed-v4.0` model produces high-quality embeddings for technical English text. It's also cost-effective compared to OpenAI embeddings. The tradeoff is an additional API dependency.

### Why Serper for web search?

Serper provides Google Search results via API with a generous free tier (2,500 queries). It's simple to integrate and returns structured results. Tavily was also considered but Serper's output format is cleaner for this use case.

### Why split /ingest into /ingest/urls and /ingest/file?

The assignment specifies a single `POST /ingest` endpoint. Splitting into two endpoints makes the API cleaner — file uploads use `multipart/form-data` while URL ingestion uses `application/json`. Combining them in one endpoint requires awkward mixed content types. Both endpoints are documented and serve the same purpose.

### Why collect feedback?

Feedback (thumbs up/down + comment) is stored in PostgreSQL. The purpose is not cosmetic — it's a data collection mechanism for future improvement. Low-rated answers identify weak points: corpus gaps, poor chunking, retrieval misses. This data can drive fine-tuning, prompt improvements, or corpus expansion. For this assignment, collection is implemented; analysis would be the next step.

### Known Limitations

- `user_id` in Streamlit resets on page refresh (no real browser cookie). In production, `extra-streamlit-components` or a proper auth system would be used.
- Hallucination checker can produce false positives on answers that synthesize context into code examples. The prompt has been tuned to reduce this but it's not perfect.
- Web search snippets are short — Serper returns brief excerpts, not full page content. This limits generation quality for web fallback answers.

---

## Chunking and Embedding Strategy

### Chunking

- **Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk size**: 800 characters
- **Overlap**: 100 characters
- **Separators**: `["\n\n", "\n", " ", ""]`

**Why 800 characters?** Technical documentation has dense information. Too small (< 300) loses context around code snippets and explanations. Too large (> 1200) dilutes the relevance signal — a chunk about many topics scores poorly against a specific query.

**Why 100 overlap?** Ensures sentences and code blocks that span chunk boundaries are captured in both adjacent chunks, preventing information loss at boundaries.

**Why RecursiveCharacterTextSplitter?** It respects document structure — tries to split on paragraphs first (`\n\n`), then lines (`\n`), then words. This keeps logical units (paragraphs, code blocks) together instead of cutting arbitrarily.

### Embeddings

- **Model**: Cohere `embed-v4.0`
- **Dimensions**: 1024
- **Distance metric**: Cosine similarity

Cohere's embedding model is optimized for semantic similarity in technical English text — well-suited for documentation retrieval. Cosine similarity is the standard for text embeddings.

---

## What I Would Improve With More Time

1. **Feedback-driven improvement** — Analyze low-rated answers to identify corpus gaps and automatically suggest missing documents to ingest.

2. **Re-ranking** — Add a cross-encoder re-ranker after retrieval to improve chunk ordering before grading. This would reduce the number of irrelevant chunks reaching the grader.

3. **Streaming responses** — Stream the LLM generation token-by-token to the Streamlit UI for better perceived performance on long answers.

4. **Persistent user_id** — Use `extra-streamlit-components` to store `user_id` in a real browser cookie so it persists across page refreshes.

5. **Evaluation pipeline** — Build a test set of question-answer pairs and run automated evals (using RAGAS or a custom evaluator) to measure retrieval precision, answer faithfulness, and hallucination rate.

6. **Async nodes** — Make LangGraph nodes async to allow parallel grading of chunks instead of sequential LLM calls.