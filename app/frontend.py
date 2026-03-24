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
    st.session_state.chat_history = []  # list of {"user": ..., "assistant": ..., "sources": [...]}

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
        docs = requests.get(f"{API_URL}/documents").json()
        if docs["count"] == 0:
            st.info("No documents indexed yet.")
        else:
            for doc in docs["documents"]:
                st.markdown(f"- 📄 `{doc}`")
    except:
        st.warning("API not reachable. Is uvicorn running?")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"💬 {len(st.session_state.chat_history)} turns in memory")

# ---------- Chat History Display ----------
st.subheader("💬 Chat")

if not st.session_state.chat_history:
    st.info("Ask a question below to start chatting with your research papers.")

for turn in st.session_state.chat_history:
    # User message
    with st.chat_message("user"):
        st.markdown(turn["user"])

    # Assistant message
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])
        # Sources in expander
        if turn.get("sources"):
            with st.expander(f"📄 {len(turn['sources'])} sources used"):
                for i, src in enumerate(turn["sources"], 1):
                    st.markdown(f"**{i}.** `{src['source']}` — Page {src['page']} | Score: `{src['score']:.3f}`")

# ---------- Input ----------
st.divider()
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is overfitting? → then ask: How do you prevent it?",
        label_visibility="collapsed"
    )
with col2:
    top_k = st.selectbox("Top-K", [3, 5, 7], index=0, label_visibility="collapsed")

ask_btn = st.button("🔍 Ask", use_container_width=True)

if ask_btn and question.strip():
    # Build history payload (exclude sources — API doesn't need them)
    history_payload = [
        {"user": t["user"], "assistant": t["assistant"]}
        for t in st.session_state.chat_history
    ]

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "top_k": top_k,
                    "chat_history": history_payload
                }
            )
            result = response.json()
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
            st.stop()

    if response.status_code == 200:
        # Append to session history
        st.session_state.chat_history.append({
            "user": question,
            "assistant": result["answer"],
            "sources": result["sources"]
        })
        st.rerun()  # refresh to show new message in chat
    else:
        st.error(f"Error: {result.get('detail', 'Unknown error')}")

elif ask_btn:
    st.warning("Please enter a question.")

# ---------- Footer ----------
st.divider()
st.caption("Built with PyMuPDF · LangChain · sentence-transformers · FAISS · Ollama · FastAPI · Streamlit")