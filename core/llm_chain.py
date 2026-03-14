# core/llm_chain.py

from typing import List, Tuple, Generator
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, LLM_MODEL
import os


# ── Prompt Templates ──────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions 
based strictly on the provided document context.

RULES:
- Answer ONLY using information from the context below
- Always cite your sources using [Source X] notation
- If the answer is not in the context, say: "I could not find this information in the provided documents."
- Never make up information or use outside knowledge
- Be concise and clear
"""

RAG_PROMPT_TEMPLATE = """
CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER (cite sources like [Source 1], [Source 2]):"""


def format_context(retrieved_docs: List[Document]) -> str:
    """Convert retrieved chunks into labeled context string."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source      = doc.metadata.get("source", "Unknown")
        page        = doc.metadata.get("page", 0)
        source_name = os.path.basename(source)
        context_parts.append(
            f"[Source {i+1}: {source_name}, Page {page + 1}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    """Convert chat history into readable string for prompt."""
    if not chat_history:
        return "No previous conversation."
    formatted = []
    for human_msg, ai_msg in chat_history:
        formatted.append(f"Human: {human_msg}")
        formatted.append(f"Assistant: {ai_msg}")
    return "\n".join(formatted)


def get_llm(streaming: bool = False) -> ChatGroq:
    """
    Returns configured Groq LLM.
    streaming=True enables token-by-token output.
    """
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
        streaming=streaming      # ← key parameter
    )


def build_prompt(
    question: str,
    retrieved_docs: List[Document],
    chat_history: List[Tuple[str, str]]
) -> str:
    """Assemble the full prompt: system + context + memory + question."""
    context     = format_context(retrieved_docs)
    history_str = format_chat_history(chat_history)
    return (
        SYSTEM_PROMPT
        + RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history_str,
            question=question
        )
    )


def generate_answer(
    question: str,
    retrieved_docs: List[Document],
    chat_history: List[Tuple[str, str]]
) -> Tuple[str, List[Document]]:
    """
    Standard (non-streaming) answer generation.
    Used by test scripts.
    """
    llm    = get_llm(streaming=False)
    prompt = build_prompt(question, retrieved_docs, chat_history)
    print(f"[Chain] Sending prompt ({len(prompt)} chars)...")
    response = llm.invoke(prompt)
    answer   = response.content
    print(f"[Chain] Answer received ({len(answer)} chars)")
    return answer, retrieved_docs


def stream_answer(
    question: str,
    retrieved_docs: List[Document],
    chat_history: List[Tuple[str, str]]
) -> Generator[str, None, None]:
    """
    Streaming answer generation.
    Yields tokens one by one as they arrive from Groq.

    Why a generator?
    Streamlit's st.write_stream() expects a generator —
    it pulls tokens from it and renders them live.
    """
    llm    = get_llm(streaming=True)
    prompt = build_prompt(question, retrieved_docs, chat_history)
    print(f"[Chain] Streaming prompt ({len(prompt)} chars)...")

    # .stream() returns chunks as they arrive
    for chunk in llm.stream(prompt):
        token = chunk.content
        if token:           # skip empty chunks
            yield token