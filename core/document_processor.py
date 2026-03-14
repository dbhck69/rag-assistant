# core/document_processor.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document                  # ← fixed
from core.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"[Loader] '{os.path.basename(file_path)}' → {len(pages)} pages loaded")
    return pages


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    all_pages = []
    for path in file_paths:
        pages = load_pdf(path)
        all_pages.extend(pages)
    print(f"[Loader] Total pages across all docs: {len(all_pages)}")
    return all_pages


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[Chunker] {len(documents)} pages → {len(chunks)} chunks")
    print(f"[Chunker] Avg chunk size: "
          f"{sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    return chunks


def process_uploaded_pdfs(file_paths: List[str]) -> List[Document]:
    raw_documents = load_multiple_pdfs(file_paths)
    chunks = chunk_documents(raw_documents)
    return chunks