# test_setup_v2.py
from dotenv import load_dotenv
import os

load_dotenv()


def test_all():
    print("=" * 50)
    print("TESTING FULL STACK")
    print("=" * 50)

    # Test 1 — Embeddings
    print("\n[1/3] Testing HuggingFace embeddings...")
    from core.embeddings import get_embedding_model
    model = get_embedding_model()
    result = model.embed_query("test sentence")
    print(f"      OK — vector size: {len(result)} dims")

    # Test 2 — FAISS
    print("\n[2/3] Testing FAISS...")
    import faiss
    print(f"      OK — FAISS version: {faiss.__version__}")

    # Test 3 — Groq
    print("\n[3/3] Testing Groq LLM...")
    from langchain_groq import ChatGroq
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("      ERROR: GROQ_API_KEY not found in .env")
        return
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=groq_key,
        temperature=0.3
    )
    response = llm.invoke("Reply with exactly: Stack verified!")
    print(f"      OK — Groq says: {response.content}")

    print("\n" + "=" * 50)
    print("ALL SYSTEMS GO — Ready for Stage 4!")
    print("=" * 50)


if __name__ == "__main__":
    test_all()