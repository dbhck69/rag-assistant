# core/retriever.py
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document              # ← fixed
from core.config import TOP_K_RESULTS


def retrieve_relevant_chunks(
    query: str,
    vector_store: FAISS,
    k: int = TOP_K_RESULTS
) -> List[Document]:
    docs = vector_store.similarity_search(query, k=k)
    print(f"[Retriever] Retrieved {len(docs)} chunks for: '{query[:60]}'")
    return docs


def retrieve_with_scores(
    query: str,
    vector_store: FAISS,
    k: int = TOP_K_RESULTS
) -> List[Tuple[Document, float]]:
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    for doc, score in docs_with_scores:
        src  = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"  Score: {score:.4f} | {src} p.{page} | "
              f"{doc.page_content[:60]}...")
    return docs_with_scores