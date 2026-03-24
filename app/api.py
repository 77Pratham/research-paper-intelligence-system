from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .query import answer_question, compare_pdfs
from .ingest import extract_text_from_pdfs, chunk_documents, embed_and_index
import shutil
import os

app = FastAPI(
    title="Research Paper Intelligence System",
    description="Local RAG pipeline — zero API cost, fully local LLM",
    version="1.0.0"
)

# Allow Streamlit frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------- Request/Response Models ----------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    chat_history: list[dict] = []  # list of {"user": ..., "assistant": ...}

class SourceItem(BaseModel):
    source: str
    page: int
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]

class CompareRequest(BaseModel):
    question: str
    pdf_names: list[str]
    top_k: int = 3


class CompareResponse(BaseModel):
    question: str
    answer: str
    compared_pdfs: list[str]
    sources: list[SourceItem]
# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"status": "running", "message": "RAG API is live"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and re-index everything in the data directory."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Re-run ingestion pipeline
    try:
        docs = extract_text_from_pdfs(DATA_DIR)
        chunks = chunk_documents(docs)
        embed_and_index(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {
        "message": f"✅ '{file.filename}' uploaded and indexed successfully",
        "total_pdfs": len(os.listdir(DATA_DIR))
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question — returns answer + source citations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = answer_question(
            request.question,
            top_k=request.top_k,
            chat_history=request.chat_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return result

@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    """Compare how multiple PDFs answer the same question."""
    if len(request.pdf_names) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 PDF names to compare")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = compare_pdfs(
            request.question,
            request.pdf_names,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    return result

@app.get("/documents")
def list_documents():
    """List all indexed PDFs."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    return {"documents": files, "count": len(files)}