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
import hashlib 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector 

# --- Pydantic Models ---
class GroundingDecision(BaseModel):
    is_general_knowledge: bool = Field(
        ..., 
        description="True if the user is asking a question that requires external, non-document knowledge, OR if the user explicitly tells the model to use its general knowledge. False if the user asks to stick strictly to the document."
    )

# --- 2. Configuration and Initialization ---
load_dotenv()
MODEL_MAPPING = {
    "fast-chat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "smart-chat": "meta-llama/Meta-Llama-3-70B-Instruct",
    "coding-expert": "codellama/CodeLlama-34b-Instruct-hf"
}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DB_URL = os.getenv("PGVECTOR_CONNECTION_STRING")
REDIS_URL = os.getenv("REDIS_URL")
deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
os.environ['OPENAI_API_KEY'] = deepinfra_key 
os.environ['OPENAI_API_BASE'] = DEEPINFRA_BASE_URL

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

# --- Health Check, Prompts, and Helper Functions ---
@app.get("/", include_in_schema=False)
def health_check(): return {"status": "ok"}

STRICT_RAG_PROMPT = """You are an expert Q&A assistant. Your ONLY source of truth is the provided CONTEXT. Answer the user's question ONLY based on the CONTEXT. If the CONTEXT does not contain the answer, state explicitly that the information is unavailable in the documents.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
strict_prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)
FLEXIBLE_RAG_PROMPT = """You are a helpful assistant. Use the provided CONTEXT first to answer the question. If the CONTEXT is insufficient or empty, use your general knowledge to provide a comprehensive answer.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
flexible_prompt = ChatPromptTemplate.from_template(FLEXIBLE_RAG_PROMPT)
PURE_CHAT_PROMPT = """You are a friendly and helpful general knowledge assistant. Use your knowledge to answer the user's question and maintain the conversation flow.

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
pure_prompt = ChatPromptTemplate.from_template(PURE_CHAT_PROMPT)

def get_llm_for_user(model_key: str):
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    return ChatDeepInfra(model=model_id, temperature=0.7)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_user_active_collection_key(user_id: str) -> str:
    return f"user:{user_id}:active_collection"

def create_and_store_vector_store(file_path: str, file_content: bytes, user_id: str):
    if EMBEDDINGS_INSTANCE is None:
        raise Exception("RAG indexing cannot proceed: Embeddings model not loaded.")
    
    file_hash = hashlib.md5(file_content).hexdigest()
    collection_name = f"user_{user_id}_doc_{file_hash}"
    logging.info(f"Creating collection: {collection_name}")

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
    os.makedirs("./temp_files", exist_ok=True)
    file_path = os.path.join("./temp_files", file.filename)
    
    file_content = await file.read()
    await file.seek(0)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if create_and_store_vector_store(file_path, file_content, current_user.user_id):
        return {"message": f"Document '{file.filename}' indexed successfully! It is now the active document for chat."}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")

@app.post("/clear_document_context")
def clear_document_context(current_user: User = Security(get_current_user)):
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
    
    history_key = f"chat:{user_id}:{request.conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    history_string = "\n".join([f"{json.loads(m.decode('utf-8'))['type']}: {json.loads(m.decode('utf-8'))['content']}" for m in history_raw])

    active_collection_key = get_user_active_collection_key(user_id)
    active_collection_name = REDIS_CLIENT_INSTANCE.get(active_collection_key)

    if active_collection_name:
        collection_name_str = active_collection_name.decode('utf-8')
        logging.info(f"User {user_id} using RAG with collection: {collection_name_str}")
        vector_store = PGVector(
            embedding_function=EMBEDDINGS_INSTANCE,
            collection_name=collection_name_str,
            connection=DB_URL
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        mode = "FLEXIBLE_RAG" 
        chat_chain = (
            {"context": retriever | format_docs, "question": itemgetter("question"), "chat_history": itemgetter("chat_history")} 
            | flexible_prompt 
            | llm_instance 
            | StrOutputParser()
        )
    else:
        logging.info(f"User {user_id} in PURE_CHAT mode.")
        mode = "PURE_CHAT"
        chat_chain = ({"question": itemgetter("question"), "chat_history": itemgetter("chat_history")} | pure_prompt | llm_instance | StrOutputParser())
    
    try:
        response_text = chat_chain.invoke({"question": request.message, "chat_history": history_string})
        history_to_save = [{"type": "human", "content": request.message}, {"type": "ai", "content": response_text}]
        for message in history_to_save:
            REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps(message))
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id, "mode": mode}
    except Exception as e:
        logging.error(f"LLM Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. Utility Endpoints ---
@app.get("/get_conversations")
def get_conversations(current_user: User = Security(get_current_user)):
    if REDIS_CLIENT_INSTANCE is None:
        raise HTTPException(status_code=500, detail="Redis connection failed.")
    
    user_id = current_user.user_id
    conversation_ids = []
    pattern = f"chat:{user_id}:*"
    for key in REDIS_CLIENT_INSTANCE.scan_iter(match=pattern):
        conv_id = key.decode('utf-8').split(':')[-1]
        conversation_ids.append(conv_id)
    return {"conversation_ids": conversation_ids}

@app.get("/history/{conversation_id}")
def get_history(conversation_id: str, current_user: User = Security(get_current_user)):
    if REDIS_CLIENT_INSTANCE is None:
         raise HTTPException(status_code=500, detail="Redis connection failed.")
         
    user_id = current_user.user_id
    history_key = f"chat:{user_id}:{conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    messages_list = [json.loads(msg.decode('utf-8')) for msg in history_raw]
    return {"conversation_id": conversation_id, "history": messages_list}