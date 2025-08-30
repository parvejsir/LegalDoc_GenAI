# modules/retriever.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from modules.embedding_store import load_faiss, INDEX_DIR

load_dotenv()

# The model name has been updated to gemini-2.0-flash
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Template: feed system prompt + context
PROMPT = """You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not contained in the context, reply exactly: Sorry it is not present in knowledge base, use google to get answer of general query.

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT)

def get_llm():
    """Create a Google Gemini LLM wrapper."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in environment.")
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )
    return llm

def answer_query(query: str, top_k: int = 4, index_path: str = str(INDEX_DIR / "faiss_index")) -> str:
    """
    Returns generated answer or the "Sorry..." message if retrieved context is insufficient.
    """
    db = load_faiss(index_path=index_path)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    # retrieve docs
    results = retriever.get_relevant_documents(query)
    
    # simple heuristic: if no results or all results are very short, then fallback
    if not results or len(" ".join([d.page_content for d in results]).strip()) < 50:
        return "Sorry it is not present in knowledge base, use google to get answer of general query"

    # combine top docs into context
    context = "\n\n---\n\n".join([d.page_content for d in results])

    # call LLM via LangChain wrapper
    llm = get_llm()
    
    # Using a more robust chain with RunnablePassthrough for better context handling
    rag_chain = prompt | llm
    
    resp = rag_chain.invoke({"context": context, "question": query}).content
    
    # If model hallucinates and returns something like "I don't know" - but we strictly want the exact phrase
    if "sorry" in resp.lower() and "knowledge base" in resp.lower():
        return "Sorry it is not present in knowledge base, use google to get answer of general query"

    return resp