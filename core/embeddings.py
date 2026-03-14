# core/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Free local embeddings using HuggingFace sentence-transformers.
    No API key needed. Runs on your CPU.
    First run downloads ~90MB model, cached forever after.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )