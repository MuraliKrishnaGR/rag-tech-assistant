"""
streamlit_app.py — Streamlit UI for RAG Technical Documentation Assistant

Features:
  - Cookie-based user_id (persists across refreshes)
  - New session_id per conversation
  - Chat interface with conversation history
  - Sidebar: past sessions, file upload, URL ingestion
  - Shows sources, hallucination warning, grade reasoning
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }

    .user-msg {
        background: #1e2130;
        border-left: 3px solid #4f8ef7;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e0e0e0;
    }

    .ai-msg {
        background: #161b27;
        border-left: 3px solid #00c896;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e0e0e0;
    }

    .warning-msg {
        background: #2d1f00;
        border-left: 3px solid #ff9800;
        padding: 10px 14px;
        border-radius: 6px;
        color: #ffb74d;
        font-size: 0.85rem;
        margin-top: 6px;
    }

    .source-tag {
        display: inline-block;
        background: #1a2744;
        color: #4f8ef7;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px 3px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None


# ── API Helpers ────────────────────────────────────────────────────────────────

def api_query(question: str) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={
                "question"  : question,
                "session_id": st.session_state.session_id,
                "user_id"   : st.session_state.user_id,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_ingest_urls(urls: list[str]) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}/ingest/urls",
            json={"urls": urls},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_ingest_file(file_bytes: bytes, filename: str) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}/ingest/file",
            files={"file": (filename, file_bytes)},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_get_user_sessions() -> list:
    try:
        response = requests.get(
            f"{API_BASE}/sessions/user/{st.session_state.user_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("sessions", [])
    except:
        return []


def api_get_session_history(session_id: str) -> list:
    try:
        response = requests.get(
            f"{API_BASE}/sessions/{session_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []


def api_delete_session(session_id: str) -> bool:
    try:
        response = requests.delete(
            f"{API_BASE}/sessions/{session_id}",
            timeout=10,
        )
        return response.status_code == 200
    except:
        return False


def api_feedback(question: str, answer: str, rating: str, comment: str = None):
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "question": question,
                "answer"  : answer,
                "rating"  : rating,
                "comment" : comment,
            },
            timeout=10,
        )
    except:
        pass


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.caption(f"User ID: `{st.session_state.user_id[:8]}...`")

    st.divider()

    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.session_id    = str(uuid.uuid4())
        st.session_state.messages      = []
        st.session_state.last_response = None
        st.rerun()

    st.divider()

    st.subheader("💬 Past Conversations")
    sessions = api_get_user_sessions()

    if not sessions:
        st.caption("No past conversations yet.")
    else:
        for session in sessions:
            sid        = session["session_id"]
            updated    = session.get("updated_at", "")[:16]
            is_current = sid == st.session_state.session_id
            label      = f"{'▶ ' if is_current else ''}{sid[:8]}... ({updated})"

            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(label, key=f"session_{sid}", use_container_width=True):
                    st.session_state.session_id = sid
                    history = api_get_session_history(sid)
                    st.session_state.messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in history
                    ]
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"delete_{sid}"):
                    api_delete_session(sid)
                    if sid == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages   = []
                    st.rerun()

    st.divider()

    st.subheader("📄 Add Documents")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "docx", "md", "txt", "html"],
    )
    if uploaded_file and st.button("Ingest File", use_container_width=True):
        with st.spinner("Ingesting file..."):
            result = api_ingest_file(uploaded_file.read(), uploaded_file.name)
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("File ingested successfully!")

    url_input = st.text_area(
        "Or enter URLs (one per line)",
        placeholder="https://docs.langchain.com/...",
        height=100,
    )
    if st.button("Ingest URLs", use_container_width=True):
        urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
        if not urls:
            st.warning("Enter at least one URL.")
        else:
            with st.spinner("Ingesting URLs..."):
                result = api_ingest_urls(urls)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"Ingested {len(urls)} URL(s) successfully!")


# ── Main Chat Area ─────────────────────────────────────────────────────────────

st.header("Ask a Question")
st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

for msg in st.session_state.messages:
    if msg["role"] == "human":
        st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

if st.session_state.last_response:
    resp = st.session_state.last_response

    if resp.get("hallucination_flag"):
        st.markdown(
            '<div class="warning-msg">⚠ This answer may not be fully grounded in the retrieved documents.</div>',
            unsafe_allow_html=True,
        )

    if resp.get("sources"):
        st.markdown("**Sources:**")
        for source in resp["sources"]:
            st.markdown(f'<span class="source-tag">🔗 {source}</span>', unsafe_allow_html=True)

    if resp.get("used_web_search"):
        st.info("🌐 This answer was generated using web search results.")

    if resp.get("grade_reasoning"):
        with st.expander("🔍 Document Grading Details"):
            for item in resp["grade_reasoning"]:
                icon = "✅" if item["grade"] == "YES" else "❌"
                st.markdown(f"{icon} **{item['source']}**")
                st.caption(f"→ {item['reason']}")

    st.markdown("**Was this answer helpful?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes", use_container_width=True):
            api_feedback(
                question=st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                answer  =st.session_state.messages[-1]["content"] if st.session_state.messages else "",
                rating  ="up",
            )
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No", use_container_width=True):
            api_feedback(
                question=st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                answer  =st.session_state.messages[-1]["content"] if st.session_state.messages else "",
                rating  ="down",
            )
            st.success("Thanks for your feedback!")

st.divider()

question = st.chat_input("Ask a question about your technical documentation...")

if question:
    st.session_state.messages.append({"role": "human", "content": question})
    st.markdown(f'<div class="user-msg">👤 {question}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        response = api_query(question)

    if "error" in response:
        st.error(f"Error: {response['error']}")
    else:
        answer = response.get("answer", "I don't have enough information.")
        st.session_state.messages.append({"role": "ai", "content": answer})
        st.markdown(f'<div class="ai-msg">🤖 {answer}</div>', unsafe_allow_html=True)
        st.session_state.last_response = response
        st.rerun()