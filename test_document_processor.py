# test_document_processor.py

from core.document_processor import process_uploaded_pdfs

# Use any PDF you have locally for testing
chunks = process_uploaded_pdfs(["C:/Users/Deepak Behera/Desktop/8th sem project/Draft Report.pdf"])

# Inspect the first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content: {chunk.page_content[:200]}")
    print(f"Metadata: {chunk.metadata}")
    print(f"Length: {len(chunk.page_content)} chars")
