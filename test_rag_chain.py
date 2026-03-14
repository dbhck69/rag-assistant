# test_rag_chain.py

from dotenv import load_dotenv
load_dotenv()

from core.document_processor import process_uploaded_pdfs
from core.vector_store import build_vector_store, save_vector_store, load_vector_store
from core.retriever import retrieve_relevant_chunks
from core.llm_chain import generate_answer
from core.memory import ConversationMemory


def test_full_rag(pdf_path: str):
    print("=" * 60)
    print("FULL RAG PIPELINE TEST")
    print("=" * 60)

    # Step 1 — Process and index the PDF
    print("\n[1/4] Processing PDF and building index...")
    chunks = process_uploaded_pdfs([pdf_path])
    vs     = build_vector_store(chunks)
    save_vector_store(vs)
    print(f"      OK — {len(chunks)} chunks indexed")

    # Step 2 — Initialize memory
    memory = ConversationMemory()
    print("\n[2/4] Memory initialized")

    # Step 3 — Ask first question
    q1 = "What is this document about? Give me a brief summary."
    print(f"\n[3/4] Question 1: {q1}")

    docs   = retrieve_relevant_chunks(q1, vs)
    answer, sources = generate_answer(q1, docs, memory.get_history())

    print(f"\nANSWER:\n{answer}")
    print(f"\nSOURCES USED:")
    for doc in sources:
        print(f"  - {doc.metadata.get('source')} "
              f"| Page {doc.metadata.get('page', 0) + 1}")

    # Save turn to memory
    memory.add_turn(q1, answer)

    # Step 4 — Ask follow-up (tests memory)
    q2 = "Can you elaborate more on the main points?"
    print(f"\n[4/4] Follow-up Question: {q2}")
    print(f"      (Memory has {len(memory)} turns — LLM knows context)")

    docs2   = retrieve_relevant_chunks(q2, vs)
    answer2, sources2 = generate_answer(q2, docs2, memory.get_history())

    print(f"\nANSWER:\n{answer2}")
    print(f"\nSOURCES USED:")
    for doc in sources2:
        print(f"  - {doc.metadata.get('source')} "
              f"| Page {doc.metadata.get('page', 0) + 1}")

    memory.add_turn(q2, answer2)

    print("\n" + "=" * 60)
    print(f"STAGE 4 COMPLETE — Memory has {len(memory)} turns")
    print("=" * 60)


if __name__ == "__main__":
    test_full_rag("Draft Report.pdf")