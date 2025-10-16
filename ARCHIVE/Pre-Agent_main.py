# main.py - FINAL STABLE VERSION

# --- 1. Dependencies and Setup ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from auth import get_current_user, User
import logging
import os
import shutil
import json
from typing import List
from operator import itemgetter
from redis import Redis
import hashlib
import psycopg
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector
from googleapiclient.discovery import build # For Google Search

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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

global EMBEDDINGS_INSTANCE
global REDIS_CLIENT_INSTANCE

try:
    REDIS_CLIENT_INSTANCE = Redis.from_url(REDIS_URL)
    REDIS_CLIENT_INSTANCE.ping()
except Exception as e:
    logging.error(f"FATAL: Redis client connection failed: {e}", exc_info=True)
    REDIS_CLIENT_INSTANCE = None

# --- 3. FastAPI Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("STARTUP: Initializing...")
    global EMBEDDINGS_INSTANCE
    try:
        EMBEDDINGS_INSTANCE = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("Embeddings model loaded.")
    except Exception as e:
        logging.error(f"FATAL: Embedding Model failed to load: {e}", exc_info=True)
        EMBEDDINGS_INSTANCE = None
    yield
    logging.info("SHUTDOWN: Application shutting down.")

app = FastAPI(lifespan=lifespan)

# --- 4. Prompt Definitions ---
@app.get("/", include_in_schema=False)
def health_check(): return {"status": "ok"}

DOCUMENT_QA_PROMPT = """You are an expert Q&A assistant for user-provided documents. Your goal is to answer the user's QUESTION using the information from the DOCUMENT CONTEXT below. If the context is insufficient, you may use your general knowledge. IMPORTANT: Do not refer to the document or the context in your response. Answer the question directly.

DOCUMENT CONTEXT:
{context}
CHAT HISTORY:
{chat_history}
QUESTION:
{question}
"""
document_qa_prompt = ChatPromptTemplate.from_template(DOCUMENT_QA_PROMPT)

PURE_CHAT_PROMPT = """You are a friendly and helpful general knowledge assistant.

CHAT HISTORY:
{chat_history}
QUESTION:
{question}
"""
pure_prompt = ChatPromptTemplate.from_template(PURE_CHAT_PROMPT)

SEARCH_PROMPT = """You are a helpful research assistant. Answer the user's QUESTION using the provided WEB SEARCH RESULTS. Summarize the key information and provide a comprehensive answer. Cite your sources by including the URL.

WEB SEARCH RESULTS:
{context}
CHAT HISTORY:
{chat_history}
QUESTION:
{question}
"""
search_prompt = ChatPromptTemplate.from_template(SEARCH_PROMPT)


# --- 5. Core Helper Functions ---
def get_llm_for_user(model_key: str):
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    return ChatDeepInfra(model=model_id, temperature=0.7)

def format_docs(docs: List[Document]) -> str:
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        if "source" in doc.metadata:
            content += f" (Source: {doc.metadata['source']})"
        formatted_docs.append(content)
    return "\n\n".join(formatted_docs)

def get_user_active_collection_key(user_id: str) -> str:
    return f"user:{user_id}:active_collection"

def get_user_active_filename_key(user_id: str) -> str:
    return f"user:{user_id}:active_filename"

def create_and_store_vector_store(file_path: str, file_content: bytes, user_id: str, original_filename: str):
    if EMBEDDINGS_INSTANCE is None:
        raise Exception("RAG indexing cannot proceed: Embeddings model not loaded.")
    file_hash = hashlib.md5(file_content).hexdigest()
    collection_name = f"user_{user_id}_doc_{file_hash}"
    logging.info(f"Creating collection: {collection_name}")
    try:
        # THIS LINE IS NOW CORRECTED
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

        # Store the collection name and the original filename in Redis
        active_collection_key = get_user_active_collection_key(user_id)
        REDIS_CLIENT_INSTANCE.set(active_collection_key, collection_name)
        
        active_filename_key = get_user_active_filename_key(user_id)
        REDIS_CLIENT_INSTANCE.set(active_filename_key, original_filename)

        return True
    except Exception as e:
        logging.error(f"RAG creation failed for collection {collection_name}: {e}", exc_info=True)
        return False

def google_search(query: str) -> List[Document]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logging.warning("Google Search credentials are not set. Skipping search.")
        return []
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()
        results = res.get("items", [])
        docs = [
            Document(page_content=item["snippet"], metadata={"source": item["link"]})
            for item in results
        ]
        logging.info(f"Google Search found {len(docs)} results for query: '{query}'")
        return docs
    except Exception as e:
        logging.error(f"Google Search failed with a specific API error: {e}", exc_info=True)
        return []

def manual_retriever(input_dict: dict) -> List[Document]:
    question = input_dict["question"]
    collection_name = input_dict["collection_name"]
    if not collection_name or EMBEDDINGS_INSTANCE is None:
        return []
    question_embedding = EMBEDDINGS_INSTANCE.embed_query(question)
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document FROM langchain_pg_embedding
                    INNER JOIN langchain_pg_collection ON langchain_pg_embedding.collection_id = langchain_pg_collection.uuid
                    WHERE langchain_pg_collection.name = %s
                    ORDER BY embedding <=> %s
                    LIMIT 3
                    """,
                    (collection_name, str(question_embedding)),
                )
                results = cur.fetchall()
                docs = [Document(page_content=row[0]) for row in results]
                logging.info(f"Manual retriever found {len(docs)} documents for collection '{collection_name}'.")
                return docs
    except Exception as e:
        logging.error(f"Manual retriever failed: {e}", exc_info=True)
        return []

# --- 6. FastAPI Endpoints ---
class ChatRequest(BaseModel):
    conversation_id: str
    model_key: str
    message: str
    use_search: bool = False

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...), current_user: User = Security(get_current_user)):
    os.makedirs("./temp_files", exist_ok=True)
    file_path = os.path.join("./temp_files", file.filename)
    file_content = await file.read()
    await file.seek(0)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    if create_and_store_vector_store(file_path, file_content, current_user.user_id, file.filename):
        return {"message": f"Document '{file.filename}' indexed. It is now the active document."}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")

@app.post("/start_new_chat")
def start_new_chat(current_user: User = Security(get_current_user)):
    """Clears the active document context for the user."""
    active_collection_key = get_user_active_collection_key(current_user.user_id)
    active_filename_key = get_user_active_filename_key(current_user.user_id)
    REDIS_CLIENT_INSTANCE.delete(active_collection_key, active_filename_key)
    return {"message": "New chat session started, document context cleared."}

@app.get("/get_active_document")
def get_active_document(current_user: User = Security(get_current_user)):
    """Checks if a user has an active document and returns its filename."""
    active_filename_key = get_user_active_filename_key(current_user.user_id)
    filename = REDIS_CLIENT_INSTANCE.get(active_filename_key)
    if filename:
        return {"active_filename": filename.decode('utf-8')}
    return {"active_filename": None}


@app.post("/chat")
async def chat_with_rag(request: ChatRequest, current_user: User = Security(get_current_user)):
    user_id = current_user.user_id
    llm_instance = get_llm_for_user(request.model_key)
    history_key = f"chat:{user_id}:{request.conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    history_string = "\n".join([f"{json.loads(m.decode('utf-8'))['type'].capitalize()}: {json.loads(m.decode('utf-8'))['content']}" for m in history_raw])
    
    active_collection_key = get_user_active_collection_key(user_id)
    active_collection_name = REDIS_CLIENT_INSTANCE.get(active_collection_key)

    if active_collection_name:
        collection_name_str = active_collection_name.decode('utf-8')
        logging.info(f"User {user_id} using RAG with collection: {collection_name_str}")
        mode = "DOCUMENT_QA"
        retriever = RunnableLambda(manual_retriever)
        chat_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
            | document_qa_prompt
            | llm_instance
            | StrOutputParser()
        )
    elif request.use_search:
        logging.info(f"User {user_id} using SEARCH mode.")
        mode = "SEARCH"
        retriever = RunnableLambda(google_search)
        chat_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
            | search_prompt
            | llm_instance
            | StrOutputParser()
        )
    else:
        logging.info(f"User {user_id} in PURE_CHAT mode.")
        mode = "PURE_CHAT"
        chat_chain = (
            {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
            | pure_prompt
            | llm_instance
            | StrOutputParser()
        )
    
    try:
        input_data = {"question": request.message, "chat_history": history_string}
        if active_collection_name:
            input_data["collection_name"] = active_collection_name.decode('utf-8')

        response_text = chat_chain.invoke(input_data)
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

