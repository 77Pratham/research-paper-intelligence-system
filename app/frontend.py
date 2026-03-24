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

# ---------- Initialize session state ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Sidebar ----------
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
        docs_response = requests.get(f"{API_URL}/documents").json()
        indexed_docs = docs_response["documents"]
        if docs_response["count"] == 0:
            st.info("No documents indexed yet.")
        else:
            for doc in indexed_docs:
                st.markdown(f"- 📄 `{doc}`")
    except:
        indexed_docs = []
        st.warning("API not reachable. Is uvicorn running?")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    st.caption(f"💬 {len(st.session_state.chat_history)} turns in memory")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["💬 Chat", "⚖️ Compare PDFs"])

# ========================
# TAB 1 — CHAT WITH MEMORY
# ========================
with tab1:
    st.subheader("💬 Chat with your research papers")

    if not st.session_state.chat_history:
        st.info("Ask a question below to start. Follow-up questions remember context!")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])
            if turn.get("sources"):
                with st.expander(f"📄 {len(turn['sources'])} sources used"):
                    for i, src in enumerate(turn["sources"], 1):
                        st.markdown(f"**{i}.** `{src['source']}` — Page {src['page']} | Score: `{src['score']:.3f}`")

    st.divider()
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is overfitting? → then ask: How do you prevent it?",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col2:
        top_k = st.selectbox("Top-K", [3, 5, 7], index=0, label_visibility="collapsed")

    if st.button("🔍 Ask", use_container_width=True, key="chat_btn"):
        if question.strip():
            history_payload = [
                {"user": t["user"], "assistant": t["assistant"]}
                for t in st.session_state.chat_history
            ]
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": question, "top_k": top_k, "chat_history": history_payload}
                    )
                    result = response.json()
                except Exception as e:
                    st.error(f"Failed to reach API: {e}")
                    st.stop()

            if response.status_code == 200:
                st.session_state.chat_history.append({
                    "user": question,
                    "assistant": result["answer"],
                    "sources": result["sources"]
                })
                st.rerun()
            else:
                st.error(f"Error: {result.get('detail', 'Unknown error')}")
        else:
            st.warning("Please enter a question.")

# ========================
# TAB 2 — MULTI-PDF COMPARE
# ========================
with tab2:
    st.subheader("⚖️ Compare across multiple PDFs")
    st.caption("Ask the same question across 2+ papers — get a structured side-by-side answer")

    if len(indexed_docs) < 2:
        st.warning("⚠️ Upload at least 2 PDFs to use comparison mode.")
    else:
        compare_question = st.text_input(
            "Comparison question",
            placeholder="e.g. How does each paper define neural networks?",
            label_visibility="collapsed",
            key="compare_input"
        )

        selected_pdfs = st.multiselect(
            "Select PDFs to compare",
            options=indexed_docs,
            default=indexed_docs[:2] if len(indexed_docs) >= 2 else indexed_docs
        )

        if st.button("⚖️ Compare", use_container_width=True, key="compare_btn"):
            if not compare_question.strip():
                st.warning("Please enter a question.")
            elif len(selected_pdfs) < 2:
                st.warning("Select at least 2 PDFs to compare.")
            else:
                with st.spinner(f"Comparing across {len(selected_pdfs)} PDFs..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/compare",
                            json={
                                "question": compare_question,
                                "pdf_names": selected_pdfs,
                                "top_k": 3
                            }
                        )
                        result = response.json()
                    except Exception as e:
                        st.error(f"Failed to reach API: {e}")
                        st.stop()

                if response.status_code == 200:
                    st.markdown("### 📊 Comparison Result")
                    st.success(result["answer"])

                    st.markdown("### 📄 Sources Used")
                    for pdf in selected_pdfs:
                        pdf_sources = [s for s in result["sources"] if s["source"] == pdf]
                        with st.expander(f"📄 {pdf} — {len(pdf_sources)} chunks used"):
                            for s in pdf_sources:
                                st.markdown(f"Page {s['page']} | Score: `{s['score']:.3f}`")
                else:
                    st.error(f"Error: {result.get('detail', 'Unknown error')}")

# ---------- Footer ----------
st.divider()
st.caption("Built with PyMuPDF · LangChain · sentence-transformers · FAISS · Ollama · FastAPI · Streamlit")