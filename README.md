# 🧠 Research Paper Intelligence System

A fully local RAG (Retrieval-Augmented Generation) pipeline that lets you query research PDFs using natural language — with zero API cost and no internet dependency.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/license-MIT-blue)

## 🏗️ Architecture
```
PDF Upload → PyMuPDF → LangChain Chunker → sentence-transformers → FAISS Index
                                                                        ↓
User Question → Embed Query → FAISS Similarity Search → Top-K Chunks
                                                                        ↓
                                          Prompt Assembly → Ollama (phi3/Llama3) → Answer + Citations
```

## ⚡ Tech Stack

| Layer | Tool | Cost |
|-------|------|------|
| PDF Ingestion | PyMuPDF | Free |
| Text Chunking | LangChain | Free |
| Embeddings | sentence-transformers | Free, local |
| Vector Store | FAISS | Free, local |
| LLM | Ollama (Llama 3 / Mistral) | Free, local |
| API | FastAPI | Free |
| Frontend | Streamlit | Free |
| CI/CD | GitHub Actions | Free tier |

**Total API cost: $0**

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/research-paper-intelligence-system.git
cd research-paper-intelligence-system
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Pull LLM model
```bash
ollama pull phi3:mini
```

### 3. Index your PDFs
Drop PDFs into the `data/` folder, then:
```bash
python app/ingest.py
```

### 4. Start the API
```bash
uvicorn app.api:app --reload --port 8000
```

### 5. Start the UI
```bash
streamlit run app/frontend.py
```

Visit `http://localhost:8501` and start querying your papers!

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload & index a PDF |
| POST | `/query` | Ask a question, get answer + citations |
| GET | `/documents` | List indexed documents |

Full docs at `http://localhost:8000/docs`

## 🎯 Key Features
- **Fully local** — no OpenAI key, no cloud, no cost
- **Source citations** — every answer links back to the exact PDF page
- **Model agnostic** — swap Ollama model with one config change
- **REST API** — queryable from any frontend or script

## ⚙️ Model Configuration

| Environment | Model | RAM Required |
|-------------|-------|-------------|
| Local dev | `phi3:mini` | ~4GB |
| Production (recommended) | `llama3` or `mistral` | ~8GB |

> The pipeline is fully model-agnostic. Swap the model with one line in `query.py`:
> `OLLAMA_MODEL = "llama3"` — no other changes needed.
> Local development uses phi3:mini due to hardware constraints.
> Production deployments should use Llama3 or Mistral for reliable multi-turn memory.

[![Watch the Demo](https://youtu.be/4jqGgtisTV8](https://youtu.be/4jqGgtisTV8)
