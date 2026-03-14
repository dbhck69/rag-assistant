
# 📚 RAG Document Assistant

### AI-Powered Multi-Document Question Answering System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-purple)

## 🎯 What is This?

A production-grade **Retrieval Augmented Generation (RAG)** system that lets you upload multiple PDF documents and have an intelligent conversation with them. Instead of relying on an LLM's training data, the system retrieves the most relevant passages from your documents and grounds every answer with citations.

> Built as a learning project to understand and implement industry-level RAG architecture from scratch.

---

## ✨ Features

- 📄 **Multi-PDF Upload** — Upload and index multiple PDFs simultaneously
- 🔍 **Semantic Search** — Find relevant content by meaning, not just keywords
- 🧠 **Conversation Memory** — Remembers last 5 turns of conversation for context
- 📌 **Source Citations** — Every answer shows exactly which document and page it came from
- ⚡ **Streaming Responses** — Answers stream word-by-word like ChatGPT
- 💾 **Persistent Index** — FAISS index saved to disk, survives app restarts
- 🆓 **100% Free Stack** — No paid APIs required for core functionality

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                                                         │
│  PDF Files → PyPDFLoader → Chunking → HuggingFace      │
│                                        Embeddings       │
│                                            ↓            │
│                                      FAISS Index        │
│                                      (persisted)        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                       │
│                                                         │
│  User Query → Embed Query → FAISS Similarity Search    │
│                                    ↓                    │
│                             Top-4 Chunks                │
│                                    ↓                    │
│              Prompt Builder (context + memory)          │
│                                    ↓                    │
│                          Groq LLaMA LLM                 │
│                                    ↓                    │
│                     Streamed Answer + Citations         │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Groq LLaMA 3.1 (free) | Answer generation |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 (local, free) | Text → vectors |
| **Vector Store** | FAISS | Semantic similarity search |
| **Document Loading** | LangChain PyPDFLoader | PDF parsing |
| **Text Splitting** | RecursiveCharacterTextSplitter | Intelligent chunking |
| **UI** | Streamlit | Chat interface |
| **Framework** | LangChain | LLM orchestration |

---

## 📁 Project Structure

```
rag_assistant/
│
├── app.py                      # Streamlit UI entry point
├── requirements.txt            # All dependencies
├── .env                        # API keys (never commit)
├── .gitignore
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── document_processor.py   # PDF loading + chunking
│   ├── embeddings.py           # HuggingFace embedding model
│   ├── vector_store.py         # FAISS operations (build/save/load)
│   ├── retriever.py            # Semantic search
│   ├── llm_chain.py            # Prompt engineering + LLM chain
│   ├── llm.py                  # LLM provider wrapper
│   └── memory.py               # Conversation history manager
│
├── data/
│   └── faiss_index/            # Persisted vector index (auto-created)
│
└── tests/
    ├── test_setup_v2.py        # Stack verification
    ├── test_vector_store.py    # FAISS pipeline test
    └── test_rag_chain.py       # End-to-end RAG test
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-assistant.git
cd rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=gsk_your_groq_key_here

# Optional — not required for core functionality
OPENAI_API_KEY=sk-proj-your_key_here
GOOGLE_API_KEY=your_google_key_here
```

Get your free Groq key at: https://console.groq.com/keys

### 5. Verify setup
```bash
python test_setup_v2.py
```

Expected output:
```
==================================================
TESTING FULL STACK
==================================================
[1/3] Testing HuggingFace embeddings...
      OK — vector size: 384 dims
[2/3] Testing FAISS...
      OK — FAISS version: 1.8.0
[3/3] Testing Groq LLM...
      OK — Groq says: Stack verified!
==================================================
ALL SYSTEMS GO — Ready for Stage 4!
==================================================
```

### 6. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 💡 How to Use

1. **Upload PDFs** — Use the sidebar to upload one or more PDF files
2. **Wait for indexing** — The system chunks and indexes your documents automatically
3. **Ask questions** — Type questions in the chat input
4. **View sources** — Expand the "Sources" panel under each answer to see citations
5. **Follow-up** — Ask follow-up questions — the system remembers conversation context

---

## ⚙️ Configuration

All settings are in `core/config.py`:

```python
# LLM
LLM_MODEL       = "llama-3.1-8b-instant"   # Groq model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # Local embedding model

# Chunking
CHUNK_SIZE    = 500    # Characters per chunk (try 200–1000)
CHUNK_OVERLAP = 50     # Overlap between chunks

# Retrieval
TOP_K_RESULTS = 4      # Chunks retrieved per query

# Storage
FAISS_INDEX_PATH = "data/faiss_index"
```

**Tuning tips:**
- Increase `CHUNK_SIZE` for longer context, decrease for more precise retrieval
- Increase `TOP_K_RESULTS` for broader answers, decrease to reduce noise
- Change `LLM_MODEL` to `llama-3.3-70b-versatile` for higher quality answers

---

## 🔑 Key Concepts Implemented

### Retrieval Augmented Generation (RAG)
Instead of relying on the LLM's training data, we retrieve relevant document passages at query time and provide them as context. This grounds answers in your actual documents and prevents hallucination.

### Semantic Search with FAISS
Documents are converted to high-dimensional vectors (embeddings) that capture meaning. When you ask a question, your question is also embedded and FAISS finds the most semantically similar document chunks — not just keyword matches.

### Chunking Strategy
Documents are split using `RecursiveCharacterTextSplitter` which tries to preserve natural language boundaries (paragraphs → sentences → words). Overlapping chunks (50 chars) ensure context isn't lost at boundaries.

### Conversation Memory
The last 5 conversation turns are injected into every prompt, enabling follow-up questions and contextual conversations across multiple turns.

---

## 🐛 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Package not installed | `pip install -r requirements.txt` |
| `groq.AuthenticationError` | Invalid API key | Check `.env` file, no quotes around key |
| `FileNotFoundError` | PDF not found | Use full absolute path or copy PDF to project root |
| FAISS loads wrong docs | Index built with different model | Delete `data/faiss_index/` and rebuild |
| Sources show temp filename | Old version of app.py | Update temp file saving logic |
| Empty source cards | HTML in expander bug | Use native Streamlit `st.markdown()` instead |

---

## 🚀 Deployment

### Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial RAG assistant"
git remote add origin https://github.com/yourusername/rag-assistant.git
git push -u origin main
```

2. Go to **https://share.streamlit.io**
3. Click **"New app"** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Add secrets in **Settings → Secrets**:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
6. Click **Deploy** — your app gets a public URL!

### Deploy on Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Add `GROQ_API_KEY` in Railway dashboard → Variables.

### Deploy on Hugging Face Spaces

1. Create a new Space at **https://huggingface.co/spaces**
2. Select **Streamlit** as the SDK
3. Upload all project files
4. Add `GROQ_API_KEY` in Settings → Repository secrets

---

## 📈 Future Improvements

- [ ] OCR support for scanned PDFs
- [ ] Support for DOCX, TXT, CSV files
- [ ] Re-ranking retrieved chunks for better precision
- [ ] Streaming with citations inline
- [ ] User authentication for multi-user deployments
- [ ] Docker containerization
- [ ] Async processing for large document sets
- [ ] Evaluation metrics (faithfulness, relevancy scores)

---

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs)
- [FAISS Documentation](https://faiss.ai)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Groq API Docs](https://console.groq.com/docs)
- [HuggingFace Sentence Transformers](https://www.sbert.net)

---

## 📄 License

MIT License — feel free to use this project for learning, portfolio, or production.

---

## 🙏 Acknowledgements

Built as part of a mentored learning project exploring Generative AI and RAG systems. Special thanks to the LangChain, FAISS, and Streamlit open-source communities.

---

*Built with Python 🐍 | LangChain 🔗 | FAISS 🔎 | Groq ⚡ | Streamlit 🎈*
=======
# rag-assistant
RAG-based multi-document Q&amp;A assistant built with LangChain, FAISS, and Streamlit. Upload PDFs, ask questions, get grounded answers with source citations. Features semantic search, conversation memory, and streaming responses. Free stack: HuggingFace embeddings + Groq LLaMA LLM
>>>>>>> 8f54442586a15fdb64bfae769d6b4b59916e8f4c
