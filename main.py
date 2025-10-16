# main.py - FINAL STABLE VERSION (With AI Search Agent)

# --- 1. Dependencies and Setup ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from pydantic import BaseModel, Field
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
from googleapiclient.discovery import build

# --- Pydantic Model for Agent's Decision ---
class SearchDecision(BaseModel):
    """The model's decision on whether to search and what to search for."""
    should_search: bool = Field(..., description="True if the user's query requires a new internet search, False otherwise.")
    query: str = Field(..., description="The optimized, concise search query to use. Should be 'None' if should_search is False.")

# --- 2. Configuration and Initialization ---
load_dotenv()
MODEL_MAPPING = {
    "fast-chat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "smart-chat": "meta-llama/Meta-Llama-3-70B-Instruct",
    "coding-expert": "codellama/CodeLlama-34b-Instruct-hf"
}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ... (rest of config is unchanged)
DB_URL = os.getenv("PGVECTOR_CONNECTION_STRING")
REDIS_URL = os.getenv("REDIS_URL")
# ... (API keys setup)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# ... (global instances and lifespan function are unchanged)
global EMBEDDINGS_INSTANCE, REDIS_CLIENT_INSTANCE

# --- 3. FastAPI Lifespan Event (omitted) ---
@asynccontextmanager
async def lifespan(app: FastAPI): #... (omitted)
    yield

app = FastAPI(lifespan=lifespan)

# --- 4. Prompt Definitions ---
@app.get("/", include_in_schema=False)
def health_check(): return {"status": "ok"}

# --- NEW: Prompt for the Planner Agent ---
PLANNER_PROMPT = """You are an expert planner agent. Your job is to determine if a new web search is required to answer the user's LATEST QUESTION.
Consider the CHAT HISTORY. If the question is a follow-up that can be answered by the history, you don't need a new search.
If the question is on a new topic, you need a search.
For example:
- User asks "What is subcritical water?". You should search.
- User then asks "What are its applications?". The previous search might have this, so you don't need a new search.
- User then asks "Can you tell me about humpback whales?". This is a new topic, so you need a new search.
- User says "Thanks!". This is conversational, you do not need a new search.

LATEST QUESTION:
{question}

CHAT HISTORY:
{chat_history}
"""
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

DOCUMENT_QA_PROMPT = """You are an expert Q&A assistant for user-provided documents... (full prompt omitted for brevity)"""
document_qa_prompt = ChatPromptTemplate.from_template(DOCUMENT_QA_PROMPT)

PURE_CHAT_PROMPT = """You are a friendly and helpful general knowledge assistant... (full prompt omitted for brevity)"""
pure_prompt = ChatPromptTemplate.from_template(PURE_CHAT_PROMPT)

SEARCH_PROMPT = """You are a helpful research assistant. Your goal is to provide a comprehensive, well-written answer to the user's QUESTION based on the provided WEB SEARCH RESULTS.
Synthesize information from all snippets. Do not just list the snippets.
Cite your sources by including the URL at the end of relevant sentences, like this [Source: www.example.com].

WEB SEARCH RESULTS:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
search_prompt = ChatPromptTemplate.from_template(SEARCH_PROMPT)


# --- 5. Core Helper Functions ---
def get_llm_for_user(model_key: str): # ... (omitted)
    return ChatDeepInfra(...)

def format_docs(docs: List[Document]) -> str: # ... (omitted)
    return "\n\n".join(...)

# ... (get_user_active... keys, create_and_store_vector_store are unchanged)

def google_search(query: str) -> List[Document]: # ... (unchanged, but added logging)
    # ...
    try:
        # ...
        logging.info(f"Google Search found {len(docs)} results for query: '{query}'")
        return docs
    except Exception as e:
        logging.error(f"Google Search failed with a specific API error: {e}", exc_info=True)
        return []

def manual_retriever(input_dict: dict) -> List[Document]: # ... (unchanged)
    # ...
    return []

# --- NEW: Planner Agent Function ---
def plan_search_or_chat(state: dict, llm_instance: ChatDeepInfra) -> SearchDecision:
    """Uses an LLM to decide whether to search the web or just chat."""
    planner_chain = planner_prompt | llm_instance.with_structured_output(SearchDecision)
    try:
        decision = planner_chain.invoke(state)
        logging.info(f"Planner decision: should_search={decision.should_search}, query='{decision.query}'")
        return decision
    except Exception as e:
        logging.error(f"Planner agent failed: {e}", exc_info=True)
        # Fallback to not searching on failure
        return SearchDecision(should_search=False, query="None")

# --- 6. FastAPI Endpoints ---
class ChatRequest(BaseModel):
    conversation_id: str
    model_key: str
    message: str
    use_search: bool = False

# ... (upload_document, start_new_chat, etc. are unchanged)

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
        # --- RAG MODE (Highest Priority) ---
        # ... (This logic is unchanged)
        mode = "DOCUMENT_QA"
        chat_chain = ...
    elif request.use_search:
        # --- SEARCH AGENT MODE ---
        # 1. Plan: Decide if a new search is needed
        planner_state = {"question": request.message, "chat_history": history_string}
        decision = plan_search_or_chat(planner_state, llm_instance)

        if decision.should_search:
            # 2. Act: Run the search and generate a response
            logging.info(f"User {user_id} executing a new search.")
            mode = "SEARCH"
            chat_chain = (
                {"context": (lambda x: decision.query) | RunnableLambda(google_search) | RunnableLambda(format_docs),
                 "question": itemgetter("question"),
                 "chat_history": itemgetter("chat_history")}
                | search_prompt
                | llm_instance
                | StrOutputParser()
            )
        else:
            # 3. Fallback: The planner decided not to search, so just chat
            logging.info(f"User {user_id} in SEARCH mode, but planner decided to just chat.")
            mode = "PURE_CHAT"
            chat_chain = (
                {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
                | pure_prompt
                | llm_instance
                | StrOutputParser()
            )
    else:
        # --- PURE CHAT MODE ---
        logging.info(f"User {user_id} in PURE_CHAT mode.")
        mode = "PURE_CHAT"
        chat_chain = (
            {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
            | pure_prompt
            | llm_instance
            | StrOutputParser()
        )

    try:
        response_text = chat_chain.invoke({"question": request.message, "chat_history": history_string})
        # ... (history saving is unchanged)
        return {"response": response_text, "mode": mode}
    except Exception as e:
        # ... (error handling is unchanged)

# --- 7. Utility Endpoints (omitted) ---
# ...

