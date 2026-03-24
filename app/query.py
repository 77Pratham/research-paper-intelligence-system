import faiss
import pickle
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer

# Paths
INDEX_FILE = "indexes/faiss.index"
CHUNKS_FILE = "indexes/chunks.pkl"

# Must match what was used in ingest.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "phi3:mini"

# Load once at module level (faster repeated queries)
print("⏳ Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("⏳ Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Index loaded — {index.ntotal} vectors ready")


def retrieve_top_chunks(question: str, top_k: int = 3) -> list[dict]:
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


def build_prompt(question: str, context_chunks: list[dict], chat_history: list[dict] = []) -> str:
    """Assemble RAG prompt — optimized for phi3:mini context limits."""
    # Truncate each chunk to 300 chars to stay within context window
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text'][:300]}"
        for c in context_chunks
    ])

    # Only include last 2 turns, truncated — phi3:mini has limited context
    history_str = ""
    if chat_history:
        history_str = "\nPrevious conversation:\n"
        for turn in chat_history[-2:]:
            history_str += f"Q: {turn['user']}\nA: {turn['assistant'][:200]}\n"
        history_str += "\n"

    return f"""Use the context below to answer the question.
If the question refers to something from the previous conversation, use that to understand it.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}
{history_str}Question: {question}
Answer:"""


def query_ollama(prompt: str) -> str:
    """Send prompt to local Ollama and return the response."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096  # explicitly set context window for phi3:mini
        }
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(question: str, top_k: int = 3, chat_history: list[dict] = []) -> dict:
    """Full pipeline: question → retrieve → prompt → LLM → answer + citations."""
    context_chunks = retrieve_top_chunks(question, top_k=top_k)
    prompt = build_prompt(question, context_chunks, chat_history)
    answer = query_ollama(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in context_chunks
        ]
    }

def retrieve_chunks_for_pdf(question: str, pdf_name: str, top_k: int = 3) -> list[dict]:
    """Retrieve top-k chunks from a SPECIFIC PDF only."""
    query_embedding = model.encode([question], convert_to_numpy=True).astype(np.float32)
    
    # Search more candidates first, then filter by PDF name
    distances, indices = index.search(query_embedding, top_k * 10)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        if chunk["source"] == pdf_name:  # filter to this PDF only
            chunk["score"] = float(dist)
            results.append(chunk)
        if len(results) >= top_k:
            break
    
    return results


def build_comparison_prompt(question: str, pdf_contexts: dict) -> str:
    """Build a structured comparison prompt across multiple PDFs."""
    sections = ""
    for pdf_name, pdf_chunks in pdf_contexts.items():
        if pdf_chunks:
            section_text = "\n".join([c["text"][:300] for c in pdf_chunks])
            sections += f"\n--- From: {pdf_name} ---\n{section_text}\n"
        else:
            sections += f"\n--- From: {pdf_name} ---\nNo relevant content found.\n"

    return f"""You are a research assistant comparing multiple documents.
Answer the question by referencing each document separately.
Structure your answer clearly with each document's perspective.

Documents:
{sections}

Question: {question}

Provide a structured comparison mentioning each document by name:"""


def compare_pdfs(question: str, pdf_names: list[str], top_k: int = 3) -> dict:
    """Compare how multiple PDFs answer the same question."""
    pdf_contexts = {}
    all_sources = []

    for pdf_name in pdf_names:
        pdf_chunks = retrieve_chunks_for_pdf(question, pdf_name, top_k=top_k)
        pdf_contexts[pdf_name] = pdf_chunks
        all_sources.extend([
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in pdf_chunks
        ])

    prompt = build_comparison_prompt(question, pdf_contexts)
    answer = query_ollama(prompt)

    return {
        "question": question,
        "answer": answer,
        "compared_pdfs": pdf_names,
        "sources": all_sources
    }

if __name__ == "__main__":
    print("\n🔍 RAG Query System Ready — type your question below")
    print("Type 'quit' to exit\n")

    conversation_history = []

    while True:
        q = input("Question: ").strip()
        if q.lower() == "quit":
            break
        if not q:
            continue

        result = answer_question(q, top_k=3, chat_history=conversation_history)

        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📄 Sources used:")
        for s in result["sources"]:
            print(f"   - {s['source']} (Page {s['page']}, score: {s['score']:.2f})")
        print()

        # Append to conversation history for next turn
        conversation_history.append({
            "user": q,
            "assistant": result["answer"]
        })