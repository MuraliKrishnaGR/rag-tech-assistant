"""
database.py — PostgreSQL connection pool + table setup
Uses psycopg2.pool.SimpleConnectionPool for efficient connection reuse.

Tables:
  - documents  → tracks ingested files/URLs with checksum for duplicate detection
  - messages   → stores full conversation history as JSON per session per user
  - feedback   → stores user feedback on answers (thumbs up/down + comment)

Note: No authentication — user_id is read from browser cookie (persists across refreshes).
      session_id is generated per conversation.
      New browser/incognito = new user_id = new cookie.
"""

import os
import json
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# ── Connection Pool ────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Neon / production
    connection_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL,
    )
else:
    # Local PostgreSQL / development
    connection_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "rag_assistant"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        port=int(os.getenv("DB_PORT", 5432)),
    )


def get_connection():
    """Get a connection from the pool."""
    return connection_pool.getconn()


def release_connection(conn):
    """Return a connection back to the pool."""
    connection_pool.putconn(conn)


# ── Table Definitions ──────────────────────────────────────────────────────────
CREATE_TABLES_SQL = """

-- Tracks every ingested document/URL
-- checksum = hash of the content we last ingested for this source_name
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    source_name  TEXT NOT NULL UNIQUE,
    source_type  TEXT NOT NULL CHECK (source_type IN ('url', 'upload')),
    checksum     TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    ingested_at  TIMESTAMP DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS messages (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    history     JSONB NOT NULL DEFAULT '[]',
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- Stores user feedback on answers
CREATE TABLE IF NOT EXISTS feedback (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    rating      TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment     TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_source_name ON documents(source_name);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
"""


def init_db():
    """Creates all tables if they don't exist. Run once at startup."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLES_SQL)
        conn.commit()
        print("[DB] Tables initialized successfully.")
    except Exception as e:
        conn.rollback()
        print(f"[DB] Error initializing tables: {e}")
        raise
    finally:
        release_connection(conn)


# ── Document Registry Helpers ──────────────────────────────────────────────────

def get_document_checksum(source_name: str) -> str | None:
    """
    Return the stored checksum for this source, or None if we've
    never ingested it before.

    Used by the indexer's fast pre-check: "has THIS source's
    content changed since we last ingested it?"
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT checksum FROM documents WHERE source_name = %s",
                (source_name,)
            )
            row = cur.fetchone()
            return row["checksum"] if row else None
    finally:
        release_connection(conn)


def register_document(source_name: str, source_type: str, checksum: str, chunk_count: int):
    """Register a brand-new source for the first time."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (source_name, source_type, checksum, chunk_count) VALUES (%s, %s, %s, %s)",
                (source_name, source_type, checksum, chunk_count)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def update_document(source_name: str, checksum: str, chunk_count: int):
    """
    Update an existing source's checksum, chunk_count, and ingested_at
    after its content has changed and been re-ingested.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET checksum = %s, chunk_count = %s, ingested_at = NOW()
                WHERE source_name = %s
                """,
                (checksum, chunk_count, source_name)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def list_documents() -> list:
    """Return all ingested documents."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, source_name, source_type, chunk_count, ingested_at FROM documents ORDER BY ingested_at DESC"
            )
            return cur.fetchall()
    finally:
        release_connection(conn)


# ── Message / Memory Helpers ───────────────────────────────────────────────────

def save_message(session_id: str, user_id: str, role: str, content: str):
    """
    Append a message to the session's JSON history.
    Creates the session row if it doesn't exist.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (session_id, user_id, history)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id) DO UPDATE
                SET history    = messages.history || %s::jsonb,
                    updated_at = NOW()
                """,
                (
                    session_id,
                    user_id,
                    json.dumps([{"role": role, "content": content}]),
                    json.dumps([{"role": role, "content": content}]),
                )
            )
        conn.commit()
    finally:
        release_connection(conn)


def get_session_messages(session_id: str) -> list[dict]:
    """
    Retrieve full chat history for a session as a Python list.
    session_id is unique — no need to filter by user_id.
    Returns: [{"role": "human", "content": "..."}, {"role": "ai", "content": "..."}]
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT history FROM messages WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return []
            return row["history"]
    finally:
        release_connection(conn)


def get_user_sessions(user_id: str) -> list[dict]:
    """
    Retrieve all sessions for a user (browser cookie).
    Useful for showing past conversations in the UI.
    Returns: [{"session_id": "...", "updated_at": "..."}]
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT session_id, updated_at FROM messages WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            return cur.fetchall()
    finally:
        release_connection(conn)
        
def delete_session(session_id: str) -> bool:
    """
    Delete a conversation session and its full history.
    Returns True if deleted, False if session not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE session_id = %s",
                (session_id,)
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        raise
    finally:
        release_connection(conn)

# ── Feedback Helpers ───────────────────────────────────────────────────────────

def save_feedback(question: str, answer: str, rating: str, comment: str = None):
    """Save user feedback on an answer."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (question, answer, rating, comment) VALUES (%s, %s, %s, %s)",
                (question, answer, rating, comment)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


if __name__ == "__main__":
    init_db()