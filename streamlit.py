# main.py
import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from modules.chunking import file_to_chunks
from modules.embedding_store import build_faiss_from_chunks, INDEX_DIR
from modules.retriever import answer_query
from modules.chatbot import init_chat, add_user_message, add_bot_message

load_dotenv()

st.set_page_config(page_title="DocChat (RAG)", layout="wide")

DATA_DIR = Path("data/uploaded_docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = str(INDEX_DIR / "faiss_index")

st.title("📚 DocChat — Ask questions from your documents")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = init_chat()
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

with st.sidebar:
    st.header("Upload & Build DB")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    build_btn = st.button("Build / Rebuild Vector DB")

    st.markdown("---")
    st.write("Embedding model:", os.getenv("EMBEDDING_MODEL"))
    st.write("LLM model:", os.getenv("LLM_MODEL"))

if uploaded is not None:
    save_path = DATA_DIR / uploaded.name
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved to {save_path}")
    st.session_state.uploaded_file = str(save_path)

if build_btn:
    if not st.session_state.uploaded_file:
        st.error("Upload a file first.")
    else:
        with st.spinner("Chunking and building embeddings..."):
            chunks = file_to_chunks(st.session_state.uploaded_file, chunk_size, chunk_overlap)
            # optional: add small metadata (filename)
            metadatas = [{"source": Path(st.session_state.uploaded_file).name, "chunk_id": i} for i in range(len(chunks))]
            db = build_faiss_from_chunks(chunks, metadatas=metadatas, index_path=INDEX_PATH)
            st.session_state.index_ready = True
            st.success(f"Built vector DB with {len(chunks)} chunks. Index saved to {INDEX_PATH}")

st.markdown("---")

# Chat UI
st.subheader("Chat")
col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input("Ask a question about uploaded documents:", key="query_input")
    if st.button("Send"):
        if not st.session_state.index_ready:
            st.error("Build the vector DB first (Sidebar).")
        elif not query:
            st.warning("Type a question.")
        else:
            add_user_message(st.session_state.chat_history, query)
            print("Query sent to LLM")
            with st.spinner("Searching documents & generating answer..."):
                answer = answer_query(query)
            add_bot_message(st.session_state.chat_history, answer)
            st.experimental_rerun()

with col2:
    st.write("Status")
    st.write("Index ready:" , st.session_state.index_ready)
    if st.session_state.uploaded_file:
        st.write("Uploaded:", Path(st.session_state.uploaded_file).name)

# show conversation
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""<div style="display:flex; gap:10px; align-items:center;">
            <img src="https://img.icons8.com/fluency/48/000000/user-male-circle.png" width="36" />
            <div style="background:#e6f2ff; padding:10px; border-radius:8px;">{st.session_state.get('user_name','User')}: {msg['text']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style="display:flex; gap:10px; align-items:center;">
            <img src="https://img.icons8.com/ios-filled/50/000000/robot-2.png" width="36" />
            <div style="background:#f1f1f1; padding:10px; border-radius:8px;">Bot: {msg['text']}</div>
            </div>""", unsafe_allow_html=True)