# main.py - FINAL VERSION with Intelligent Query Router

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
import asyncio
import requests
import hashlib
from typing import List, Optional, Literal
from operator import itemgetter
from redis import Redis
import psycopg
from bs4 import BeautifulSoup

# LangChain and AI Model Imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector
from tavily import TavilyClient

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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

global EMBEDDINGS_INSTANCE, REDIS_CLIENT_INSTANCE
try:
    REDIS_CLIENT_INSTANCE = Redis.from_url(REDIS_URL)
    REDIS_CLIENT_INSTANCE.ping()
except Exception as e:
    logging.error(f"FATAL: Redis client connection failed: {e}", exc_info=True)
    REDIS_CLIENT_INSTANCE = None

# --- 3. FastAPI Lifespan & Health Check ---
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
@app.get("/", include_in_schema=False)
def health_check(): return {"status": "ok"}

# --- 4. Prompt Definitions ---
PURE_CHAT_PROMPT = ChatPromptTemplate.from_template("""You are a friendly and helpful general knowledge assistant.

CHAT HISTORY:
{chat_history}
QUESTION:
{question}
""")

RESEARCH_AGENT_PROMPT = ChatPromptTemplate.from_template("""You are an expert-level research analyst. Your primary goal is to produce a high-quality, comprehensive, and unbiased summary of the user's QUESTION. You must base your answer *exclusively* on the provided WEB SEARCH RESULTS.

**Critical Instructions:**
1.  **Synthesize, Don't List:** Weave the information together into a single, well-written narrative summary. Use paragraphs to structure your answer.
2.  **Cite Everything:** This is the most important rule. **Every single sentence** you write must be followed by a citation to the source that supports it, like this: [Source: URL]. If a sentence is supported by multiple sources, cite them all, like this: [Source: URL1, URL2].
3.  **No Outside Information:** Do not add any information, context, or opinions that are not explicitly found in the provided text.
4.  **Handle Contradictions:** If sources conflict, point this out directly in your summary.

**WEB SEARCH RESULTS:**
{context}

**QUESTION:**
{question}
""")

ROUTER_PROMPT = ChatPromptTemplate.from_template("""You are an expert query classifier. Your task is to analyze the user's LATEST QUESTION in the context of the CHAT HISTORY and determine the best way to answer it.

Classify the LATEST QUESTION into one of three categories:
1.  `RESEARCH_REQUIRED`: The question is complex, seeks real-time information (recent events, news), requires deep specialist knowledge, or explicitly asks for research.
2.  `GENERAL_KNOWLEDGE`: The question can be answered using general knowledge and does not require a web search. This includes definitions, historical facts, or simple explanations.
3.  `CONVERSATIONAL`: The question is a simple conversational follow-up, a greeting, a thank you, or an instruction that doesn't require new information.

CHAT HISTORY:
{chat_history}

LATEST QUESTION:
{question}

Respond with a JSON object containing your classification. Example: {{"query_type": "RESEARCH_REQUIRED"}}""")

# Other prompts (DOCUMENT_QA_PROMPT) remain the same...
DOCUMENT_QA_PROMPT = ChatPromptTemplate.from_template("""(Your original DOCUMENT_QA_PROMPT here)""")


# --- 5. Core Helper and Agent Functions ---
def get_llm_for_user(model_key: str):
    # ... (function remains the same)
    pass

def format_docs(docs: List[Document]) -> str:
    # ... (function remains the same)
    pass

# ... (user active key functions, create_and_store_vector_store, manual_retriever remain the same)

async def scrape_url(url: str) -> Optional[Document]:
    # ... (function remains the same)
    pass

async def scrape_and_filter_urls(urls: List[str]) -> List[Document]:
    # ... (function remains the same)
    pass

async def run_tavily_research_agent(query: str, llm_instance) -> List[Document]:
    logging.info("--- Running Tavily Research Agent ---")
    if not TAVILY_API_KEY:
        logging.error("Tavily API key not set. Search will fail.")
        return []
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        search_results = await asyncio.to_thread(tavily_client.search, query=query, search_depth="advanced", max_results=5)
        unique_urls = [item['url'] for item in search_results.get('results', [])]
        logging.info(f"Tavily Agent found {len(unique_urls)} URLs.")
        return await scrape_and_filter_urls(unique_urls)
    except Exception as e:
        logging.error(f"Tavily search failed: {e}", exc_info=True)
        return []

# 🧠 NEW: The Intelligent Query Router
class RouterResponse(BaseModel):
    query_type: Literal["RESEARCH_REQUIRED", "GENERAL_KNOWLEDGE", "CONVERSATIONAL"] = Field(
        description="The classification of the user's query."
    )

async def route_query(question: str, chat_history: str) -> str:
    logging.info(f"Routing query: {question}")
    router_llm = get_llm_for_user("fast-chat") # Use the cheapest model for routing
    parser = JsonOutputParser(pydantic_object=RouterResponse)
    
    router_chain = ROUTER_PROMPT | router_llm | parser
    
    try:
        result = await router_chain.ainvoke({"question": question, "chat_history": chat_history})
        logging.info(f"Query classified as: {result['query_type']}")
        return result['query_type']
    except Exception as e:
        logging.error(f"Router failed: {e}. Defaulting to GENERAL_KNOWLEDGE.")
        return "GENERAL_KNOWLEDGE"


# --- 6. FastAPI Endpoints ---
class ChatRequest(BaseModel):
    conversation_id: str
    model_key: str
    message: str
    use_search: bool = False # This toggle now enables the router

@app.post("/chat")
async def chat_with_rag(request: ChatRequest, current_user: User = Security(get_current_user)):
    user_id = current_user.user_id
    llm_instance = get_llm_for_user(request.model_key)
    synthesis_llm = get_llm_for_user("smart-chat")
    
    history_key = f"chat:{user_id}:{request.conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    history_string = "\n".join([json.loads(m.decode('utf-8'))['content'] for m in history_raw])

    active_collection_key = get_user_active_collection_key(user_id)
    active_collection_name = REDIS_CLIENT_INSTANCE.get(active_collection_key)
    
    response_text = ""
    mode = ""

    if active_collection_name:
        # Document RAG logic takes priority and remains the same
        # ...
        pass
    elif request.use_search:
        # If search is enabled, use the router to decide the path
        query_type = await route_query(request.message, history_string)
        
        if query_type == "RESEARCH_REQUIRED":
            mode = "RESEARCH_AGENT (Tavily)"
            documents = await run_tavily_research_agent(request.message, llm_instance)
            if not documents:
                response_text = "I tried to search for that, but I couldn't find enough information online."
            else:
                context_string = format_docs(documents)
                synthesis_chain = RESEARCH_AGENT_PROMPT | synthesis_llm | StrOutputParser()
                response_text = await synthesis_chain.ainvoke({"context": context_string, "question": request.message})
        else: # GENERAL_KNOWLEDGE or CONVERSATIONAL
            mode = f"PURE_CHAT ({query_type})"
            chat_chain = PURE_CHAT_PROMPT | llm_instance | StrOutputParser()
            response_text = await chat_chain.ainvoke({"question": request.message, "chat_history": history_string})
    else:
        # Default behavior if search is disabled
        mode = "PURE_CHAT (Search Disabled)"
        chat_chain = PURE_CHAT_PROMPT | llm_instance | StrOutputParser()
        response_text = await chat_chain.ainvoke({"question": request.message, "chat_history": history_string})

    # Save history and return response
    try:
        history_to_save = [{"type": "human", "content": request.message}, {"type": "ai", "content": response_text}]
        for message in history_to_save:
            REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps(message))
        
        final_model_used = synthesis_llm.model_name if "RESEARCH" in mode else llm_instance.model_name
        return {
            "response": response_text, 
            "model_used": final_model_used, 
            "conversation_id": request.conversation_id, 
            "mode": mode
        }
    except Exception as e:
        logging.error(f"Final response stage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ... (All other endpoints like /upload_document, /get_conversations etc. remain unchanged)