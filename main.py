# main.py - FINAL PRODUCTION CODE (Multi-User RAG Logic)

# --- 1. Dependencies and Setup ---
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
import hashlib # NEW: For creating unique document IDs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector 

# --- Pydantic Models (omitted for brevity) ---
class GroundingDecision(BaseModel):
    is_general_knowledge: bool = Field(...)

# --- 2. Configuration and Initialization (omitted for brevity) ---
load_dotenv()
MODEL_MAPPING = {"fast-chat": "...", "smart-chat": "...", "coding-expert": "..."}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DB_URL = os.getenv("PGVECTOR_CONNECTION_STRING")
REDIS_URL = os.getenv("REDIS_URL")
# ... (DeepInfra setup) ...

global EMBEDDINGS_INSTANCE
global REDIS_CLIENT_INSTANCE
# ... (Redis client initialization) ...

# --- 3. FastAPI Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("STARTUP: Initializing...")
    global EMBEDDINGS_INSTANCE
    try:
        EMBEDDINGS_INSTANCE = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("Embeddings model loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL: Embedding Model failed to load: {e}", exc_info=True)
        EMBEDDINGS_INSTANCE = None 
    yield 
    logging.info("SHUTDOWN: Application shutting down.")

app = FastAPI(lifespan=lifespan)

# --- Health Check, Prompts, and Helper Functions (omitted for brevity) ---
@app.get("/", include_in_schema=False)
def health_check(): return {"status": "ok"}
STRICT_RAG_PROMPT = """..."""
strict_prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)
# ... (other prompts and helpers) ...
def get_llm_for_user(model_key: str): #...
    return ChatDeepInfra(...)
# ...

# --- MODIFIED: RAG Helper Functions ---
def get_user_active_collection_key(user_id: str) -> str:
    """Returns the Redis key that stores the user's current active collection name."""
    return f"user:{user_id}:active_collection"

def create_and_store_vector_store(file_path: str, file_content: bytes, user_id: str):
    """
    Creates a unique vector store for a document and user, and sets it as the user's active collection.
    """
    if EMBEDDINGS_INSTANCE is None:
        raise Exception("RAG indexing cannot proceed: Embeddings model not loaded.")

    # 1. Create a unique, repeatable collection name for this specific document
    file_hash = hashlib.md5(file_content).hexdigest()
    collection_name = f"user_{user_id}_doc_{file_hash}"
    logging.info(f"Creating collection: {collection_name}")

    # 2. Load, split, and index the document into the new collection
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        PGVector.from_documents(
            documents=splits, 
            embedding=EMBEDDINGS_INSTANCE, 
            collection_name=collection_name,
            connection=DB_URL
        )
        
        # 3. CRITICAL: Store this new collection name in Redis as the user's active one
        active_collection_key = get_user_active_collection_key(user_id)
        REDIS_CLIENT_INSTANCE.set(active_collection_key, collection_name)
        
        return True
    except Exception as e:
        logging.error(f"RAG creation failed for collection {collection_name}: {e}", exc_info=True)
        return False

# --- 6. FastAPI Endpoints ---

class ChatRequest(BaseModel):
    conversation_id: str
    model_key: str
    message: str

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...), current_user: User = Security(get_current_user)):
    """Accepts a file, saves it, and creates a user-specific RAG index."""
    os.makedirs("./temp_files", exist_ok=True)
    file_path = os.path.join("./temp_files", file.filename)
    
    # Read file content to pass for hashing
    file_content = await file.read()
    await file.seek(0) # Reset file pointer after reading

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if create_and_store_vector_store(file_path, file_content, current_user.user_id):
        return {"message": f"Document '{file.filename}' indexed successfully! It is now the active document for chat."}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")

@app.post("/clear_document_context")
def clear_document_context(current_user: User = Security(get_current_user)):
    """Clears the active document context for the user, returning them to PURE_CHAT mode."""
    active_collection_key = get_user_active_collection_key(current_user.user_id)
    deleted_count = REDIS_CLIENT_INSTANCE.delete(active_collection_key)
    
    if deleted_count > 0:
        return {"message": "Document context cleared successfully. You are now in normal chat mode."}
    else:
        return {"message": "No active document context to clear."}

@app.post("/chat")
async def chat_with_rag(request: ChatRequest, current_user: User = Security(get_current_user)):
    user_id = current_user.user_id
    llm_instance = get_llm_for_user(request.model_key)
    
    # 1. Retrieve chat history from Redis (same as before)
    history_key = f"chat:{user_id}:{request.conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    history_string = "\n".join([f"{json.loads(m)['type']}: {json.loads(m)['content']}" for m in history_raw])

    # 2. NEW: Check if the user has an active RAG collection in Redis
    active_collection_key = get_user_active_collection_key(user_id)
    active_collection_name = REDIS_CLIENT_INSTANCE.get(active_collection_key)

    if active_collection_name:
        # --- RAG MODE ---
        # Decode the collection name from bytes to string
        collection_name_str = active_collection_name.decode('utf-8')
        logging.info(f"User {user_id} using RAG with collection: {collection_name_str}")

        # Instantiate a PGVector store on-the-fly for this specific request
        vector_store = PGVector(
            embedding_function=EMBEDDINGS_INSTANCE,
            collection_name=collection_name_str,
            connection=DB_URL
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # ... RAG chain logic (same as before) ...
        selected_prompt = strict_prompt # or flexible_prompt
        mode = "STRICT_RAG"
        chat_chain = ({"context": retriever, "question": RunnablePassthrough(), "chat_history": lambda x: history_string} | selected_prompt | llm_instance | StrOutputParser())
    else:
        # --- PURE CHAT MODE ---
        logging.info(f"User {user_id} in PURE_CHAT mode.")
        mode = "PURE_CHAT"
        chat_chain = ({"question": RunnablePassthrough(), "chat_history": lambda x: history_string} | pure_prompt | llm_instance | StrOutputParser())
    
    # 3. Generation & History Update (same as before)
    try:
        response_text = chat_chain.invoke(request.message)
        # ... save history to Redis ...
        return {"response": response_text, "mode": mode}
    except Exception as e:
        # ... error handling ...

# --- 7. Utility Endpoints (omitted for brevity) ---
@app.get("/get_conversations")
def get_conversations(current_user: User = Security(get_current_user)): #...
    return {"conversation_ids": []}

@app.get("/history/{conversation_id}")
def get_history(conversation_id: str, current_user: User = Security(get_current_user)): #...
    return {"history": []}