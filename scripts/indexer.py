import os
import sys
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    BSHTMLLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import vectorstore
from db import get_document_checksum, register_document, update_document


# ── Chunking Strategy ─────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)


# ── Default LangChain doc URLs ────────────────────────────────────
DEFAULT_URLS = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/quickstart",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph",
]

# ── Supported File Types ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_from_urls(urls: list[str]) -> list[Document]:
    """Scrape and load documents from a list of URLs."""
    print(f"\n[URL] Loading {len(urls)} URL(s)...")
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = url
                doc.metadata["source_type"] = "url"
            docs.extend(loaded)
            print(f"  ✓ {url}")
        except Exception as e:
            print(f"  ✗ Failed to load {url}: {e}")
    return docs


def load_from_file_bytes(file_bytes: bytes, filename: str) -> list[Document]:
    """Load a file from raw bytes. Supported: .pdf, .docx, .md, .txt, .html"""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    print(f"\n[FILE] Loading '{filename}' ({ext})...")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif ext in {".md", ".txt"}:
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif ext == ".html":
            loader = BSHTMLLoader(tmp_path)

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["source_type"] = "upload"

        print(f"  ✓ Loaded {len(docs)} page(s) from '{filename}'")
        return docs
    finally:
        os.unlink(tmp_path)


# ── Checksum Helpers ──────────────────────────────────────────────────────────

def chunk_checksum(text: str) -> str:
    """SHA-256 of a single chunk's raw text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_checksum(docs: list[Document]) -> str:
    """SHA-256 of the entire document's concatenated text.
    Used as a fast pre-check before doing chunk-level comparison.
    """
    full_text = "\n".join(doc.page_content for doc in docs)
    return hashlib.sha256(full_text.encode("utf-8")).hexdigest()


# ── Chroma Helpers ────────────────────────────────────────────────────────────

def get_existing_chunks(source_name: str) -> dict[str, str]:
    """
    Fetch all existing Chroma chunks for a source.
    Returns: {chunk_checksum: chroma_id}
    """
    collection = vectorstore._collection
    results = collection.get(
        where={"source": source_name},
        include=["metadatas"],
    )

    existing = {}
    for chroma_id, metadata in zip(results["ids"], results["metadatas"]):
        cs = metadata.get("chunk_checksum")
        if cs:
            existing[cs] = chroma_id

    return existing


def delete_chroma_ids(ids: list[str]):
    """Delete specific chunk IDs from Chroma."""
    if ids:
        vectorstore._collection.delete(ids=ids)
        print(f"  [Chroma] Deleted {len(ids)} stale chunk(s)")


# ── Core Dedup + Ingest ───────────────────────────────────────────────────────

def ingest_incremental(docs: list[Document], source_name: str, source_type: str,) -> dict:
    """
    Chunk-level dedup ingestion, per source.

    Steps:
      1. Fast pre-check: whole-document checksum vs. the checksum we stored
         for THIS source last time → if same, skip immediately (zero Chroma calls)
      2. Get existing chunk checksums from Chroma for this source
      3. Compute new chunk checksums
      4. Diff: find new chunks to add, stale chunk IDs to delete
      5. Embed only new chunks (Cohere API call only for these)
      6. Delete stale chunks from Chroma
      7. Sync Postgres: insert if new source, update if existing source
    """
    doc_cs = document_checksum(docs)

    # ── Step 1: Fast per-source pre-check ────────────────────────────────────
    previous_cs = get_document_checksum(source_name)
    if previous_cs == doc_cs:
        print(f"  [Dedup] SKIP — '{source_name}' content unchanged")
        return {"status": "skipped", "source": source_name, "chunks_added": 0, "chunks_deleted": 0}

    # ── Step 2: Get existing chunk checksums from Chroma ─────────────────────
    existing_chunks = get_existing_chunks(source_name)
    # existing_chunks = {chunk_checksum: chroma_id, ...}
    is_new_source = previous_cs is None

    # ── Step 3: Split + stamp new chunks ─────────────────────────────────────
    new_chunks = text_splitter.split_documents(docs)
    for chunk in new_chunks:
        chunk.metadata["chunk_checksum"] = chunk_checksum(chunk.page_content)

    new_checksums = {
        chunk.metadata["chunk_checksum"]: chunk
        for chunk in new_chunks
    }

    # ── Step 4: Diff ──────────────────────────────────────────────────────────
    # Chunks in new content but not in Chroma → need embedding
    to_add = [
        chunk for cs, chunk in new_checksums.items()
        if cs not in existing_chunks
    ]

    # Chunks in Chroma but not in new content → stale, delete
    to_delete_ids = [
        chroma_id for cs, chroma_id in existing_chunks.items()
        if cs not in new_checksums
    ]

    # ── Step 5: Embed only new chunks ─────────────────────────────────────────
    if to_add:
        vectorstore.add_documents(to_add)
        print(f"  [Chroma] Embedded {len(to_add)} new chunk(s) for '{source_name}'")
    else:
        print(f"  [Chroma] No new chunks to embed for '{source_name}'")

    # ── Step 6: Delete stale chunks ───────────────────────────────────────────
    delete_chroma_ids(to_delete_ids)

    # ── Step 7: Sync Postgres ─────────────────────────────────────────────────
    if is_new_source:
        register_document(source_name, source_type, doc_cs, len(new_chunks))
        status = "ingested"
    else:
        update_document(source_name, doc_cs, len(new_chunks))
        status = "updated"

    return {
        "status": status,
        "source": source_name,
        "chunks_added": len(to_add),
        "chunks_deleted": len(to_delete_ids),
        "chunks_total": len(new_chunks),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_urls(urls: list[str]) -> list[dict]:
    """Load + dedup + ingest a list of URLs."""
    results = []
    docs_by_url: dict[str, list[Document]] = {}

    raw_docs = load_from_urls(urls)

    # Group by URL — WebBaseLoader may return multiple docs per URL
    for doc in raw_docs:
        url = doc.metadata.get("source", "unknown")
        docs_by_url.setdefault(url, []).append(doc)

    for url, docs in docs_by_url.items():
        result = ingest_incremental(docs, source_name=url, source_type="url")
        results.append(result)
        print(f"  → {result}")

    return results


def ingest_file(file_bytes: bytes, filename: str) -> dict:
    """Load + dedup + ingest an uploaded file."""
    docs = load_from_file_bytes(file_bytes, filename)
    result = ingest_incremental(docs, source_name=filename, source_type="upload")
    print(f"  → {result}")
    return result


# ── Startup: seed default URLs ────────────────────────────────────────────────

def ingest_default_urls():
    """Called once at app startup to ensure default docs are indexed."""
    print("\n[Startup] Checking default URLs...")
    results = ingest_urls(DEFAULT_URLS)
    skipped = sum(1 for r in results if r["status"] == "skipped")
    updated = sum(1 for r in results if r["status"] == "updated")
    ingested = sum(1 for r in results if r["status"] == "ingested")
    print(f"\n[Startup] Done — {ingested} new, {updated} updated, {skipped} unchanged")