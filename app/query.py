import faiss
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Paths
INDEX_FILE = "indexes/faiss.index"
CHUNKS_FILE = "indexes/chunks.pkl"

# Must match what was used in ingest.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

# Load once at module level (faster repeated queries)
print("⏳ Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("⏳ Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Index loaded — {index.ntotal} vectors ready")


def retrieve_top_chunks(question: str, top_k: int = 5) -> list[dict]:
    """Embed the question and find top-k similar chunks from FAISS."""
    query_embedding = model.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            chunk = chunks[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)
    return results


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Assemble the RAG prompt from question + retrieved chunks."""
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    ])
    return f"""You are a research assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def query_ollama(prompt: str) -> str:
    """Send prompt to local Ollama and stream back the response."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(question: str, top_k: int = 5) -> dict:
    """Full pipeline: question → retrieve → prompt → LLM → answer + citations."""
    context_chunks = retrieve_top_chunks(question, top_k=top_k)
    prompt = build_prompt(question, context_chunks)
    answer = query_ollama(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in context_chunks
        ]
    }


if __name__ == "__main__":
    print("\n🔍 RAG Query System Ready — type your question below")
    print("Type 'quit' to exit\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() == "quit":
            break
        if not q:
            continue
        result = answer_question(q)
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📄 Sources used:")
        for s in result["sources"]:
            print(f"   - {s['source']} (Page {s['page']}, score: {s['score']:.2f})")
        print()