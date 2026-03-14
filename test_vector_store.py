# test_vector_store.py
from dotenv import load_dotenv
import os

load_dotenv()

from core.document_processor import process_uploaded_pdfs
from core.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store
)
from core.retriever import retrieve_relevant_chunks, retrieve_with_scores
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GOOGLE_API_KEY


def test_pipeline(pdf_path: str, test_query: str):

    # Step 1 — Process PDF
    print("\n--- Step 1: Processing PDF ---")
    chunks = process_uploaded_pdfs([pdf_path])

    # Step 2 — Build FAISS index
    print("\n--- Step 2: Building vector store ---")
    vs = build_vector_store(chunks)

    # Step 3 — Save to disk
    print("\n--- Step 3: Saving index ---")
    save_vector_store(vs)

    # Step 4 — Reload from disk
    print("\n--- Step 4: Loading index from disk ---")
    vs_loaded = load_vector_store()

    # Step 5 — Retrieve with scores
    print(f"\n--- Step 5: Querying: '{test_query}' ---")
    results = retrieve_with_scores(test_query, vs_loaded)

    print("\n--- Top Results ---")
    for doc, score in results:
        print(f"\nScore : {score:.4f}")
        print(f"Source: {doc.metadata.get('source')} "
              f"| Page {doc.metadata.get('page', 0) + 1}")
        print(f"Text  : {doc.page_content[:200]}")


def test_gemini():
    print("\n--- Testing Gemini connection ---")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    response = llm.invoke("Say: Gemini connection successful!")
    print(f"Gemini says: {response.content}")


if __name__ == "__main__":
    test_pipeline(
        pdf_path="Draft Report.pdf",
        test_query="What is the main topic of this document?"
    )
    test_gemini()
