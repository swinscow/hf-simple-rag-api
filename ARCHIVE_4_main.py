# main.py - FINAL PRODUCTION CODE (Stable Cloud Configuration)

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
# from langchain_community.chat_message_histories import RedisChatMessageHistory 
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

# Initialize Redis client globally (non-blocking)
try:
    REDIS_CLIENT_INSTANCE = Redis.from_url(REDIS_URL)
    REDIS_CLIENT_INSTANCE.ping()
except Exception as e:
    logging.error(f"FATAL: Redis client connection failed during global init: {e}", exc_info=True)
    REDIS_CLIENT_INSTANCE = None


# --- 3. FastAPI Lifespan Event (The Cloud Stability Fix) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("STARTUP: Running initialization...")
    
    global vector_store
    global EMBEDDINGS_INSTANCE
    
    vector_store = None 

    try:
        # Load the memory-intensive embedding model 
        EMBEDDINGS_INSTANCE = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("DB Status: Embeddings model loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL: Embedding Model failed to load: {e}", exc_info=True)
        EMBEDDINGS_INSTANCE = None 

    yield # Application ready!

    logging.info("SHUTDOWN: Application shutting down.")


app = FastAPI(lifespan=lifespan) # <--- FINAL APP DECLARATION

# --- New Simple Health Check Endpoint ---
@app.get("/", include_in_schema=False)
def health_check():
    """Render's internal health check."""
    return {"status": "ok", "db_initialized": vector_store is not None}


# --- 4. Prompt Definitions ---
STRICT_RAG_PROMPT = """
You are an expert Q&A assistant. Your ONLY source of truth is the provided CONTEXT. 
Answer the user's question ONLY based on the CONTEXT. 
If the CONTEXT does not contain the answer, state explicitly that the information is unavailable in the documents.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
strict_prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)

FLEXIBLE_RAG_PROMPT = """
You are a helpful assistant. Use the provided CONTEXT first to answer the question.
If the CONTEXT is insufficient or empty, use your general knowledge to provide a comprehensive answer.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
flexible_prompt = ChatPromptTemplate.from_template(FLEXIBLE_RAG_PROMPT)

PURE_CHAT_PROMPT = """
You are a friendly and helpful general knowledge assistant. Use your knowledge to answer the user's question and maintain the conversation flow.

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
pure_prompt = ChatPromptTemplate.from_template(PURE_CHAT_PROMPT)


# --- 5. Core Helper Functions ---

def get_llm_for_user(model_key: str):
    """Dynamically creates the LLM instance based on user's model choice."""
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    return ChatDeepInfra(
        model=model_id,
        temperature=0.7,
    )

def check_for_general_intent(message: str, llm_instance: ChatDeepInfra) -> bool:
    """Uses the LLM to classify whether the user intends to use general knowledge."""
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an Intent Classifier. Your task is to determine if the user intends "
         "to use general knowledge (True) or strictly restrict the answer to a document (False). "
         "Output ONLY the JSON object. Default to True unless the message strictly implies restriction."
         "Example: 'Only use the file', 'Strictly answer from the document', 'Stick to the context'."
        ),
        ("user", "User Message: {message}")
    ])
    
    classification_chain = (
        classification_prompt
        | llm_instance.with_structured_output(GroundingDecision)
    )

    try:
        decision = classification_chain.invoke({"message": message})
        return not decision.is_general_knowledge
    except Exception as e:
        logging.error(f"LLM classification failed: {e}", exc_info=True)
        # Fallback to keyword search if the LLM classification fails
        return any(keyword in message.lower() for keyword in ["only use the file", "only based on the file", "only use the document", "strictly only"])

def create_vector_store(file_path: str):
    """Loads, chunks, embeds, and indexes the document into the PGVector database."""
    global vector_store

    if DB_ENGINE is None:
        raise Exception("RAG indexing cannot proceed: Database Engine initialization failed during startup.")

    try:
        # CRITICAL FIX: Use the global embedding instance loaded in the lifespan
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Load and split documents
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Use the stable .from_documents method (Handles table creation/connection)
        vector_store = PGVector.from_documents(
            documents=splits, 
            embedding=embeddings, 
            collection_name=COLLECTION_NAME,
            connection=DB_URL # Use the URL string for PGVector.from_documents
        )
        
        return True
    except Exception as e:
        logging.error(f"RAG creation failed during runtime: {e}", exc_info=True)
        return False

def format_docs(docs):
    """Converts a list of documents into a single string for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 6. FastAPI Endpoints ---

class ChatRequest(BaseModel):
    conversation_id: str = "new_chat"
    model_key: str = "fast-chat"
    message: str


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a file, saves it, and creates the local RAG index."""
    os.makedirs("./temp_files", exist_ok=True)
    file_path = os.path.join("./temp_files", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if create_vector_store(file_path):
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully! Use the /chat endpoint now."}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")


# main.py - FINAL AND CORRECT CHAT ENDPOINT WITH PURE REDIS MEMORY

@app.post("/chat")
async def chat_with_rag(
    request: ChatRequest, 
    current_user: User = Security(get_current_user) 
):
    global vector_store
    global REDIS_CLIENT_INSTANCE

    # --- SECURITY BYPASS: Set user_id for testing (MUST BE REMOVED LATER) ---
    user_id = current_user.user_id
    
    # 1. Initialization and Retrieval
    session_id = f"{user_id}_{request.conversation_id}"

    # Check for Redis client availability (Fatal if not connected)
    if REDIS_CLIENT_INSTANCE is None:
         raise HTTPException(status_code=500, detail="Chat history unavailable. Redis connection failed on startup.")

    # Define the key used to store this conversation in Redis
    history_key = f"chat:{user_id}:{request.conversation_id}"

    # Retrieve history directly from Redis (avoids crashing wrapper)
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    
    # Deserialize the history into a usable string format
    history_messages = [json.loads(msg.decode('utf-8')) for msg in history_raw]
    history_string = "\n".join([f"{msg['type'].capitalize()}: {msg['content']}" for msg in history_messages])

    llm_instance = get_llm_for_user(request.model_key)
    
    # 2. Logic Branching (RAG / PURE CHAT)
    if vector_store is not None:
        # --- RAG MODE (Document Available) ---
        is_strict = check_for_general_intent(request.message, llm_instance)
        selected_prompt = strict_prompt if is_strict else flexible_prompt
        mode = "STRICT_RAG" if is_strict else "FLEXIBLE_RAG"
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        chat_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": itemgetter("question") | retriever | format_docs 
            }
            | selected_prompt 
            | llm_instance
            | StrOutputParser()
        )

    else:
        # --- PURE CHAT MODE (No Document Indexed) ---
        mode = "PURE_CHAT"
        
        chat_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | pure_prompt
            | llm_instance
            | StrOutputParser()
        )
    
    # 3. Generation & History Update
    try:
        input_dict = {
            "question": request.message,
            "chat_history": history_string,
        }
        
        response_text = chat_chain.invoke(input_dict)
        
        # --- CRITICAL: Save the new messages directly to Redis (STABLE LOGIC) ---
        history_to_save = [
            {"type": "human", "content": request.message},
            {"type": "ai", "content": response_text}
        ]
        for message in history_to_save:
            # RPUSH adds the message to the end of the list key
            REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps(message))
        # ----------------------------------------------------------------------
        
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id, "mode": mode}
        
    except Exception as e:
        logging.error(f"LLM Inference Detail Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")

# --- 7. Utility Endpoints (Cleanup & History Retrieval) ---

@app.get("/get_conversations")
def get_conversations(current_user: User = Security(get_current_user)):
    """
    Returns a list of all saved conversation IDs for the logged-in user.
    NOTE: This is a placeholder for the MVP, as scanning Redis is complex.
    """
    user_id = current_user.user_id
    
    return {
        "user_id": user_id,
        "warning": "Conversation history listing requires scanning Redis, which is complex for an MVP.",
        "suggestion": "Proceed with building the Streamlit UI to manage sessions via the URL.",
    }


@app.get("/history/{conversation_id}")
def get_history(
    conversation_id: str, 
    current_user: User = Security(get_current_user)
):
    """Debug endpoint to view the persistent Redis chat history."""
    global REDIS_CLIENT_INSTANCE

    user_id = current_user.user_id
    session_id = f"{user_id}_{conversation_id}"
    
    if REDIS_CLIENT_INSTANCE is None:
         raise HTTPException(status_code=500, detail="History unavailable. Redis connection failed.")
         
    # CRITICAL: We initialize a *new* history manager for read access
    history_manager = RedisChatMessageHistory(
        session_id=session_id, 
        key_prefix=user_id,
        client=REDIS_CLIENT_INSTANCE
    )
    
    messages_list = [{"type": msg.type, "content": msg.content} for msg in history_manager.messages]
    
    return {"session_id": session_id, "history": messages_list}