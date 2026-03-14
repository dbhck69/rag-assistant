# app.py — Streamlit RAG Assistant with Streaming

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from core.document_processor import process_uploaded_pdfs
from core.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    add_documents_to_store
)
from core.retriever import retrieve_relevant_chunks
from core.llm_chain import stream_answer, generate_answer
from core.memory import ConversationMemory


# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


# ── Session state ─────────────────────────────────────────
if "vector_store"    not in st.session_state:
    st.session_state.vector_store    = None
if "memory"          not in st.session_state:
    st.session_state.memory          = ConversationMemory()
if "chat_history"    not in st.session_state:
    st.session_state.chat_history    = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "total_chunks"    not in st.session_state:
    st.session_state.total_chunks    = 0


def render_sources(sources: list):
    """
    Render source citations using pure Streamlit components.
    No unsafe_allow_html needed — works reliably in all contexts.
    """
    for i, src in enumerate(sources):
        st.markdown(f"**📌 Source {i+1}: {src['file']} — Page {src['page']}**")
        st.caption(src["snippet"])
        if i < len(sources) - 1:
            st.divider()


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")

    # Status indicator
    if st.session_state.vector_store:
        st.success(f"🟢 Ready — {st.session_state.total_chunks} chunks indexed")
    else:
        st.warning("🔴 No documents loaded")

    st.markdown("---")
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to chat with"
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]

        if new_files:
            progress   = st.progress(0, text="Starting...")
            temp_paths = []

            for i, uploaded_file in enumerate(new_files):
                progress.progress(
                    int((i / len(new_files)) * 50),
                    text=f"Reading {uploaded_file.name}..."
                )
                # Save to temp dir using ORIGINAL filename
                # so metadata shows correct name in citations
                temp_dir  = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.read())
                temp_paths.append((uploaded_file.name, temp_file))

            progress.progress(50, text="Chunking documents...")
            all_paths = [p for _, p in temp_paths]
            chunks    = process_uploaded_pdfs(all_paths)

            progress.progress(75, text="Building vector index...")
            if st.session_state.vector_store is None:
                st.session_state.vector_store = build_vector_store(chunks)
            else:
                st.session_state.vector_store = add_documents_to_store(
                    st.session_state.vector_store, chunks
                )

            save_vector_store(st.session_state.vector_store)

            progress.progress(95, text="Finalizing...")
            st.session_state.total_chunks += len(chunks)
            for name, _ in temp_paths:
                st.session_state.processed_files.append(name)
            import shutil
            for _, path in temp_paths:
                shutil.rmtree(os.path.dirname(path), ignore_errors=True)

            progress.progress(100, text="Done!")
            st.success(
                f"✅ {len(new_files)} file(s) processed — "
                f"{len(chunks)} chunks added"
            )

    # Indexed files list
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("📋 Indexed Files")
        for fname in st.session_state.processed_files:
            st.markdown(f"📄 `{fname}`")

    # Controls
    st.markdown("---")
    st.subheader("⚙️ Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.vector_store    = None
            st.session_state.memory          = ConversationMemory()
            st.session_state.chat_history    = []
            st.session_state.processed_files = []
            st.session_state.total_chunks    = 0
            st.rerun()

    st.markdown("---")
    st.caption(f"🧠 Memory: {len(st.session_state.memory)} turns")
    st.caption(f"💬 Messages: {len(st.session_state.chat_history)}")


# ── Main area ─────────────────────────────────────────────
st.title("💬 Document Q&A Assistant")
st.caption("Powered by Groq LLaMA + FAISS + HuggingFace Embeddings")

# Welcome screen
if not st.session_state.processed_files:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nUpload PDF files using the sidebar")
    with col2:
        st.info("**Step 2**\n\nWait for indexing to complete")
    with col3:
        st.info("**Step 3**\n\nAsk questions about your documents")
    st.markdown("---")

# Render existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Render sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(
                f"📄 Sources ({len(message['sources'])} used)"
            ):
                render_sources(message["sources"])


# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.processed_files
):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })

    # Generate streaming response
    with st.chat_message("assistant"):
        try:
            # Retrieve chunks
            docs = retrieve_relevant_chunks(
                prompt,
                st.session_state.vector_store
            )

            # Stream answer word by word
            full_answer = st.write_stream(
                stream_answer(
                    prompt,
                    docs,
                    st.session_state.memory.get_history()
                )
            )

            # Build deduplicated source list
            sources = []
            seen    = set()
            for doc in docs:
                fname = os.path.basename(
                    doc.metadata.get("source", "Unknown")
                )
                page = doc.metadata.get("page", 0) + 1
                key  = (fname, page)
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file":    fname,
                        "page":    page,
                        "snippet": doc.page_content[:250] + "..."
                    })

            # Show sources using native Streamlit
            if sources:
                with st.expander(
                    f"📄 Sources ({len(sources)} used)"
                ):
                    render_sources(sources)

        except Exception as e:
            full_answer = f"Sorry, I encountered an error: {str(e)}"
            st.error(full_answer)
            sources = []

    # Save to memory and history
    st.session_state.memory.add_turn(prompt, full_answer)
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources
    })