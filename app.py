# app.py

import os
import uuid
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables FIRST
load_dotenv()

# Get the API key right after loading the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the key is None and raise an error early
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")

# Now, import your modules that rely on the key
from summarizer import generate_document_summary
from models import SummaryResponse, LegalDocSummary
from utils import extract_text_from_pdf
from modules.chunking import file_to_chunks
from modules.embedding_store import build_faiss_from_chunks, INDEX_DIR
from modules.retriever import answer_query
from modules.chatbot import init_chat, add_user_message, add_bot_message

# Global state for the application
db_session: Dict[str, dict] = {}

# Initialize FastAPI
app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the chat request body
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str

# --- Chat Endpoints ---

@app.post("/upload-and-build/")
async def upload_and_build_db(file: UploadFile = File(...)):
    """
    Handles file upload, chunks the document, and builds the FAISS index.
    """
    global db_session
    
    filepath = Path("data") / file.filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = file_to_chunks(str(filepath))
        if not chunks:
            raise HTTPException(status_code=400, detail="The document is empty or could not be processed.")
        
        metadatas = [{"source": file.filename, "chunk_id": i} for i in range(len(chunks))]
        db = build_faiss_from_chunks(chunks, metadatas=metadatas, index_path=str(INDEX_DIR))
        
        conversation_id = str(uuid.uuid4())
        db_session[conversation_id] = {
            "faiss_db": db,
            "chat_history": init_chat()
        }

        return {
            "message": f"Successfully processed {len(chunks)} chunks and built the vector DB.",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    finally:
        os.remove(filepath)

@app.post("/chat/")
async def chat_with_docs(request: ChatRequest):
    """
    Handles the chat query, retrieves relevant context, and gets an answer.
    """
    conversation_id = request.conversation_id
    query = request.query
    
    if not conversation_id or conversation_id not in db_session:
        raise HTTPException(status_code=400, detail="Invalid or missing conversation ID. Please upload a document first.")
    
    session_data = db_session[conversation_id]
    
    add_user_message(session_data["chat_history"], query)
    
    try:
        answer = answer_query(query, index_path=str(INDEX_DIR))
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Vector DB not found. Please upload a document first.")
    
    add_bot_message(session_data["chat_history"], answer)
    
    db_session[conversation_id]["chat_history"] = session_data["chat_history"]
    
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "chat_history": session_data["chat_history"]
    }

# --- Summarizer Endpoint ---

@app.post("/summarize/", response_model=SummaryResponse)
async def summarize_document(file: UploadFile = File(...), language: str = Form(...)):
    """
    Handles document summarization request.
    This endpoint always generates a new summary on each call.
    """
    file_location = f"temp/{file.filename}"
    Path(file_location).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        document_content = extract_text_from_pdf(file_location)
        if not document_content:
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        # Pass the API key explicitly to the summarization function
        summary_result: LegalDocSummary = generate_document_summary(document_content, language, GOOGLE_API_KEY)
        print(type(summary_result.json))
        return SummaryResponse(
            summary=summary_result,
            is_summarized=False # Always return false as we are not using a cache
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # Ensure temporary file is always cleaned up
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)