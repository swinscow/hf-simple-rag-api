# main.py - FINAL PRODUCTION-READY API CODE (Cloud/Supabase/Auth0)

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
from typing import Dict, List, Any
from operator import itemgetter 

# CRITICAL HACK TO BYPASS STUBBORN SSL HANDSHAKE FAILURES
# This disables verification for ALL HTTPS requests made by Python
import ssl
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
# END CRITICAL HACK

# LangChain Components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector, PostgresChatMessageHistory 


# --- Pydantic Model for Intent Classification ---
class GroundingDecision(BaseModel):
    """Model to force the LLM to output a structured decision."""
    is_general_knowledge: bool = Field(
        ..., 
        description="True if the user is asking a question that requires external, non-document knowledge, OR if the user explicitly tells the model to use its general knowledge. False if the user asks to stick strictly to the document."
    )

# --- 2. Configuration and Initialization ---
load_dotenv()

# --- Model Definitions (User-Facing Name -> Deep Infra ID) ---
MODEL_MAPPING = {
    "fast-chat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "smart-chat": "meta-llama/Meta-Llama-3-70B-Instruct",
    "coding-expert": "codellama/CodeLlama-34b-Instruct-hf"
}

# Configuration Variables
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
COLLECTION_NAME = "rag_documents"

# CRITICAL: Database connection setup (Reads PGVECTOR_CONNECTION_STRING from Render ENV)
DB_URL = os.getenv("PGVECTOR_CONNECTION_STRING")
if not DB_URL:
    raise ValueError("DB_URL (PGVECTOR_CONNECTION_STRING) not found. Check Render ENV!")

# CRITICAL: Deep Infra Authentication Setup
deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
if not deepinfra_key:
    raise ValueError("DEEPINFRA_API_KEY not found. Check Render ENV!")

# Set environment variables for LangChain compatibility
os.environ['OPENAI_API_KEY'] = deepinfra_key 
os.environ['OPENAI_API_BASE'] = DEEPINFRA_BASE_URL

# Initialize DB connection pool engine (non-blocking)
try:
    DB_ENGINE = create_engine(DB_URL.replace("+psycopg", "")) 
except Exception as e:
    logging.critical(f"FATAL: Database Engine creation failed: {e}")
    DB_ENGINE = None

vector_store = None 


# --- 3. FastAPI Lifespan Event (The Cloud Crash Fix) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("STARTUP: Application starting...")
    
    global vector_store
    vector_store = None # Ensure global placeholder is set

    try:
        # NOTE: We load the embedding model once here to satisfy the vector_store requirement 
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL) 
        
        # We attempt to initialize PGVector by connecting to the existing table/collection
        # This will fail gracefully if the table doesn't exist, which is handled in /upload_document
        from langchain_postgres.vectorstores import PGVector
        
        # We need the vector store to be defined so the /chat endpoint can use the retriever
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DB_URL
        )
        logging.info("DB Status: PGVector placeholder initialized successfully.")

    except Exception as e:
        # This catches errors like connection failure during startup, preventing a crash
        logging.error(f"FATAL: PGVector initialization failed during startup. Indexing disabled until upload. Error: {e}")
        vector_store = None # Ensure it is None if it failed

    # SERVER IS READY
    yield 

    logging.info("SHUTDOWN: Application shutting down.")


app = FastAPI(lifespan=lifespan) # <--- FINAL APP DECLARATION (MUST BE UNINDENTED)

# --- New Simple Health Check Endpoint ---
@app.get("/", include_in_schema=False)
def health_check():
    """Render's internal health check."""
    return {"status": "ok", "db_initialized": vector_store is not None}


# --- 4. Prompt Definitions ---
# (Prompts for STRICT, FLEXIBLE, and PURE chat modes)
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

    if vector_store is not None:
        # CRITICAL: This is the first time the PGVector store is created.
        # This will create the tables in Supabase.
        logging.info("Re-initializing PGVector store for new indexing.")
        try:
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

            # 1. Load and split documents
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # 2. Use the stable .from_documents method (Handles table creation/connection)
            vector_store = PGVector.from_documents(
                documents=splits, 
                embedding=embeddings, 
                collection_name=COLLECTION_NAME,
                connection=DB_URL
            )
            logging.info("RAG Indexing SUCCESS: Documents added to PGVector.")
            return True
        except Exception as e:
            logging.error(f"RAG creation failed during runtime: {e}", exc_info=True)
            return False
    else:
        # This path should ideally not be hit after the lifespan event is run, but serves as a safety check
        raise Exception("RAG indexing cannot proceed: Vector store was not initialized during startup.")


def format_docs(docs):
    """Converts a list of documents into a single string for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 7. FastAPI Endpoints ---

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
    
    # CRITICAL: The create_vector_store function must run successfully here
    if create_vector_store(file_path):
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully! Use the /chat endpoint now."}
    else:
        # RAG creation failed during runtime. Return a 500 error.
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")


@app.post("/chat")
async def chat_with_rag(
    request: ChatRequest, 
    current_user: User = Security(get_current_user) 
):
    global vector_store
    
    user_id = current_user.user_id 

    # 1. Initialization and History Retrieval
    session_id = f"{user_id}_{request.conversation_id}"
    history_manager = PostgresChatMessageHistory(connection_string=DB_URL, session_id=session_id)
    history_messages = history_manager.messages
    
    history_string = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history_messages])
    llm_instance = get_llm_for_user(request.model_key)
    
    # 2. Logic Branching
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
        
        # Update the history store with the new messages
        history_manager.add_user_message(request.message)
        history_manager.add_ai_message(response_text)
        
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id, "mode": mode}
        
    except Exception as e:
        logging.error(f"LLM Inference Detail Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")


# --- 7. Utility Endpoints ---

@app.get("/get_conversations")
def get_conversations(current_user: User = Security(get_current_user)):
    """
    Returns a list of all saved conversation IDs for the logged-in user.
    NOTE: This is a placeholder for the MVP, as scanning the DB is complex.
    """
    user_id = current_user.user_id
    
    return {
        "user_id": user_id,
        "warning": "Conversation history listing is complex and requires a direct DB query.",
        "suggestion": f"Log into your Supabase Dashboard and check the 'langchain_chat_history' table for sessions starting with: {user_id}_",
    }


@app.get("/history/{conversation_id}")
def get_history(
    conversation_id: str, 
    current_user: User = Security(get_current_user)
):
    """Debug endpoint to view the persistent PostgreSQL chat history."""
    user_id = current_user.user_id
    session_id = f"{user_id}_{conversation_id}"
    
    history_manager = PostgresChatMessageHistory(connection_string=DB_URL, session_id=session_id)
    
    messages_list = [{"type": msg.type, "content": msg.content} for msg in history_manager.messages]
    
    return {"session_id": session_id, "history": messages_list}