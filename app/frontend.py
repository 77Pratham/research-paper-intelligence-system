import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Research Paper Intelligence System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Research Paper Intelligence System")
st.caption("Fully local RAG pipeline — zero API cost | phi3:mini via Ollama")

# ---------- Sidebar — Upload PDFs ----------
with st.sidebar:
    st.header("📂 Upload Research Papers")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Indexing PDF..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Upload failed: {response.text}")

    st.divider()
    st.header("📄 Indexed Documents")

    try:
        docs = requests.get(f"{API_URL}/documents").json()
        if docs["count"] == 0:
            st.info("No documents indexed yet.")
        else:
            for doc in docs["documents"]:
                st.markdown(f"- 📄 `{doc}`")
    except:
        st.warning("API not reachable. Is uvicorn running?")

# ---------- Main — Query ----------
st.divider()
st.subheader("💬 Ask a Question")

col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input("", placeholder="e.g. What is supervised learning?", label_visibility="collapsed")
with col2:
    top_k = st.selectbox("Top-K chunks", [3, 5, 7], index=1)

ask_btn = st.button("🔍 Ask", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k}
            )
            result = response.json()
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
            st.stop()

    if response.status_code == 200:
        # Answer
        st.markdown("### 💡 Answer")
        st.success(result["answer"])

        # Citations
        st.markdown("### 📄 Sources Used")
        for i, src in enumerate(result["sources"], 1):
            with st.expander(f"Source {i} — {src['source']} | Page {src['page']} | Score: {src['score']:.3f}"):
                st.markdown(f"**File:** `{src['source']}`")
                st.markdown(f"**Page:** {src['page']}")
                st.markdown(f"**Similarity Score:** `{src['score']:.4f}`")
    else:
        st.error(f"Error: {result.get('detail', 'Unknown error')}")

elif ask_btn:
    st.warning("Please enter a question first.")

# ---------- Footer ----------
st.divider()
st.caption("Built with PyMuPDF · LangChain · sentence-transformers · FAISS · Ollama · FastAPI · Streamlit")