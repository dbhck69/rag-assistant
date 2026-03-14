# core/llm.py
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, LLM_MODEL


def get_llm():
    """
    Returns the LLM instance.
    Isolated here so swapping providers means
    changing only this one file.
    """
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )