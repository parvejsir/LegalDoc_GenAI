# modules/embedding_store.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = Path("vectorstore/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def get_embedding_model():
    """Get the Hugging Face embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_faiss_from_chunks(chunks, metadatas=None, index_path: str = str(INDEX_DIR / "faiss_index")):
    embedding = get_embedding_model()
    docs = []
    for i, chunk in enumerate(chunks):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {"chunk_id": i}
        docs.append(Document(page_content=chunk, metadata=meta))
    db = FAISS.from_documents(docs, embedding)
    db.save_local(index_path)
    return db

def load_faiss(index_path: str = str(INDEX_DIR / "faiss_index")):
    """Load the FAISS index."""
    embedding = get_embedding_model()
    idx_path = Path(index_path)
    if not idx_path.exists():
        raise FileNotFoundError(f"No index at {index_path}")
    
    db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    return db