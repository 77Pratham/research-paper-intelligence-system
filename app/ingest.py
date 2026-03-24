import os
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "data"
INDEX_DIR = "indexes"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.pkl")

# Model (downloads once, runs locally forever after)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_pdfs(data_dir: str) -> list[dict]:
    """Extract raw text from all PDFs in the data directory."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # skip blank pages
                    documents.append({
                        "text": text,
                        "source": filename,
                        "page": page_num + 1
                    })
            print(f"✅ Extracted: {filename} ({len(doc)} pages)")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,  # overlap keeps context from being cut off
        length_function=len
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "source": doc["source"],
                "page": doc["page"]
            })
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def embed_and_index(chunks: list[dict]):
    """Embed chunks and save FAISS index to disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("⏳ Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    print(f"⏳ Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Save index and chunks metadata
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ FAISS index saved → {INDEX_FILE}")
    print(f"✅ Chunks metadata saved → {CHUNKS_FILE}")


if __name__ == "__main__":
    print("🚀 Starting ingestion pipeline...")
    docs = extract_text_from_pdfs(DATA_DIR)
    if not docs:
        print("⚠️  No PDFs found in /data. Drop some PDFs in and retry.")
    else:
        chunks = chunk_documents(docs)
        embed_and_index(chunks)
        print("🎉 Ingestion complete! Ready to query.")