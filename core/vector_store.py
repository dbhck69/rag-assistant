# core/vector_store.py
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document              # ← fixed
from core.embeddings import get_embedding_model
from core.config import FAISS_INDEX_PATH, TOP_K_RESULTS


def build_vector_store(chunks: List[Document]) -> FAISS:
    print(f"[VectorStore] Embedding {len(chunks)} chunks...")
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    print(f"[VectorStore] FAISS index built successfully.")
    return vector_store


def save_vector_store(vector_store: FAISS) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"[VectorStore] Index saved to '{FAISS_INDEX_PATH}'")


def load_vector_store() -> Optional[FAISS]:
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if not os.path.exists(index_file):
        print("[VectorStore] No saved index found.")
        return None
    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print(f"[VectorStore] Index loaded from '{FAISS_INDEX_PATH}'")
    return vector_store


def add_documents_to_store(
    vector_store: FAISS,
    new_chunks: List[Document]
) -> FAISS:
    print(f"[VectorStore] Adding {len(new_chunks)} new chunks...")
    vector_store.add_documents(new_chunks)
    print("[VectorStore] Chunks added successfully.")
    return vector_store