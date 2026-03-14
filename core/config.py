# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ── LLM settings ──────────────────────────────────────────
LLM_PROVIDER    = "groq"
LLM_MODEL       = "llama-3.1-8b-instant"   # ← updated model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking settings ─────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval settings ────────────────────────────────────
TOP_K_RESULTS = 4

# ── Paths ─────────────────────────────────────────────────
FAISS_INDEX_PATH = "data/faiss_index"