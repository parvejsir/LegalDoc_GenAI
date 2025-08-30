# modules/chunking.py
from pathlib import Path
from typing import List
import docx
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

SUPPORTED = (".pdf", ".txt", ".docx")

def load_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_docx(path: Path) -> str:
    doc = docx.Document(path)
    full = []
    for para in doc.paragraphs:
        full.append(para.text)
    return "\n".join(full)

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    full = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full.append(text)
    return "\n".join(full)

def load_file_to_text(filepath: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext == ".txt":
        return load_txt(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".pdf":
        return load_pdf(path)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

def file_to_chunks(filepath: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    text = load_file_to_text(filepath)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks