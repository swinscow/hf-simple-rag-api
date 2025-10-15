# main.py - FINAL PRODUCTION CODE (With Conversation History Logic)

# --- 1. Dependencies and Setup (CRITICAL: All imports at the top) ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from pydantic import BaseModel, Field
from dotenv import load_dotenv 
from contextlib import asynccontextmanager 
from sqlalchemy import create_engine
from auth import get_current_user, User 
import logging
import os
import shutil
import json
from typing import Dict, List, Any
from operator import itemgetter 
from redis import Redis 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector 


# --- Pydantic Model for Intent Classification ---
class GroundingDecision(BaseModel):
    """Model to force the LLM to output a structured decision."""
    is_general_knowledge: bool = Field(
        ..., 
        description="True if the user is asking a question that requires external, non-document knowledge, OR if the user explicitly tells the model to use its general knowledge. False if the user asks to stick strictly to the document."
    )

# --- 2. Configuration and Initialization ---
load_dotenv()

# --- Model Definitions ---
MODEL_MAPPING = {
    "fast-chat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "smart-chat": "meta-llama/Meta-Llama-3-70B-Instruct",
    "coding-expert": "codellama/CodeLlama-34b-Instruct-hf"
}

# Configuration Variables
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
COLLECTION_NAME = "rag_documents"

# CRITICAL: Environment Variables
DB_URL = os.getenv("PGVECTOR_CONNECTION_STRING")
REDIS_URL = os.getenv("REDIS_URL")
if not DB_URL or not REDIS_URL:
    raise ValueError("Missing critical DB or REDIS URL in Render ENV!")
    
# Deep Infra Authentication Setup
deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
if not deepinfra_key:
    raise ValueError("DEEPINFRA_API_KEY not found. Check Render ENV!")
os.environ['OPENAI_API_KEY'] = deepinfra_key 
os.environ['OPENAI_API_BASE'] = DEEPINFRA_BASE_URL

# Initialize Global Instances
try:
    DB_ENGINE = create_engine(DB_URL.replace("+psycopg", "")) 
except Exception as e:
    logging.critical(f"FATAL: Database Engine creation failed: {e}", exc_info=True)
    DB_ENGINE = None

vector_store = None 
global EMBEDDINGS_INSTANCE
global REDIS_CLIENT_INSTANCE

try:
    REDIS_CLIENT_INSTANCE = Redis.from_url(REDIS_URL)
    REDIS_CLIENT_INSTANCE.ping()
except Exception as e:
    logging.error(f"FATAL: Redis client connection failed during global init: {e}", exc_info=True)
    REDIS_CLIENT_INSTANCE = None


# --- 3. FastAPI Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("STARTUP: Running initialization...")
    global vector_store
    global EMBEDDINGS_INSTANCE
    vector_store = None 
    try:
        EMBEDDINGS_INSTANCE = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("DB Status: Embeddings model loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL: Embedding Model failed to load: {e}", exc_info=True)
        EMBEDDINGS_INSTANCE = None 
    yield 
    logging.info("SHUTDOWN: Application shutting down.")


app = FastAPI(lifespan=lifespan) 

# --- Health Check Endpoint ---
@app.get("/", include_in_schema=False)
def health_check():
    return {"status": "ok", "db_initialized": vector_store is not None}


# --- 4. Prompt Definitions ---
STRICT_RAG_PROMPT = """You are an expert Q&A assistant... (omitted for brevity)"""
strict_prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)
FLEXIBLE_RAG_PROMPT = """You are a helpful assistant... (omitted for brevity)"""
flexible_prompt = ChatPromptTemplate.from_template(FLEXIBLE_RAG_PROMPT)
PURE_CHAT_PROMPT = """You are a friendly and helpful assistant... (omitted for brevity)"""
pure_prompt = ChatPromptTemplate.from_template(PURE_CHAT_PROMPT)


# --- 5. Core Helper Functions ---
def get_llm_for_user(model_key: str): # ... (omitted for brevity)
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    return ChatDeepInfra(model=model_id, temperature=0.7)

def check_for_general_intent(message: str, llm_instance: ChatDeepInfra) -> bool: # ... (omitted for brevity)
    # ...
    return any(keyword in message.lower() for keyword in ["only use the file", "strictly only"])

def create_vector_store(file_path: str): # ... (omitted for brevity)
    global vector_store
    # ...
    return True

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- 6. FastAPI Endpoints ---
class ChatRequest(BaseModel):
    conversation_id: str = "new_chat"
    model_key: str = "fast-chat"
    message: str


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...), current_user: User = Security(get_current_user)):
    os.makedirs("./temp_files", exist_ok=True)
    file_path = os.path.join("./temp_files", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    if create_vector_store(file_path):
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully!"}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")


@app.post("/chat")
async def chat_with_rag(request: ChatRequest, current_user: User = Security(get_current_user)):
    global vector_store, REDIS_CLIENT_INSTANCE
    user_id = current_user.user_id
    if REDIS_CLIENT_INSTANCE is None:
         raise HTTPException(status_code=500, detail="Redis connection failed.")
    history_key = f"chat:{user_id}:{request.conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    history_messages = [json.loads(msg.decode('utf-8')) for msg in history_raw]
    history_string = "\n".join([f"{msg['type'].capitalize()}: {msg['content']}" for msg in history_messages])
    llm_instance = get_llm_for_user(request.model_key)
    
    if vector_store is not None:
        is_strict = check_for_general_intent(request.message, llm_instance)
        selected_prompt = strict_prompt if is_strict else flexible_prompt
        mode = "STRICT_RAG" if is_strict else "FLEXIBLE_RAG"
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        chat_chain = ({"question": itemgetter("question"), "chat_history": itemgetter("chat_history"), "context": itemgetter("question") | retriever | format_docs} | selected_prompt | llm_instance | StrOutputParser())
    else:
        mode = "PURE_CHAT"
        chat_chain = ({"question": itemgetter("question"), "chat_history": itemgetter("chat_history")} | pure_prompt | llm_instance | StrOutputParser())
    
    try:
        response_text = chat_chain.invoke({"question": request.message, "chat_history": history_string})
        history_to_save = [{"type": "human", "content": request.message}, {"type": "ai", "content": response_text}]
        for message in history_to_save:
            REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps(message))
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id, "mode": mode}
    except Exception as e:
        logging.error(f"LLM Inference Detail Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")

# --- 7. Utility Endpoints (Cleanup & History Retrieval) ---

# --- NEW: Implemented /get_conversations endpoint ---
@app.get("/get_conversations")
def get_conversations(current_user: User = Security(get_current_user)):
    """
    Scans Redis for all conversation keys for the logged-in user and returns the IDs.
    """
    if REDIS_CLIENT_INSTANCE is None:
        raise HTTPException(status_code=500, detail="History unavailable. Redis connection failed.")
    
    user_id = current_user.user_id
    conversation_ids = []
    
    # Use SCAN to safely find all keys matching the user's chat history pattern
    pattern = f"chat:{user_id}:*"
    for key in REDIS_CLIENT_INSTANCE.scan_iter(match=pattern):
        # The key is bytes, so we decode it. Then split to get the conversation_id part.
        conv_id = key.decode('utf-8').split(':')[-1]
        conversation_ids.append(conv_id)
        
    return {"conversation_ids": conversation_ids}


# --- NEW: Refactored /history endpoint for stability ---
@app.get("/history/{conversation_id}")
def get_history(conversation_id: str, current_user: User = Security(get_current_user)):
    """
    Retrieves the chat history for a specific conversation using the stable direct Redis method.
    """
    if REDIS_CLIENT_INSTANCE is None:
         raise HTTPException(status_code=500, detail="History unavailable. Redis connection failed.")
         
    user_id = current_user.user_id
    history_key = f"chat:{user_id}:{conversation_id}"
    
    # Use the same stable method as the /chat endpoint
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    
    # Deserialize the history into a list of dictionaries
    messages_list = [json.loads(msg.decode('utf-8')) for msg in history_raw]
    
    return {"conversation_id": conversation_id, "history": messages_list}