# main.py - FINAL DEFINITIVE VERSION (Dynamic Strategy Generation)

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
from typing import List, Optional, Union
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
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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


# --- 4. Pydantic Models for Structured Output ---
class SearchStrategy(BaseModel):
    requires_search: bool = Field(description="Is a search required to answer the query?")
    topic: str = Field(description="The core subject or topic of the user's query.")
    location: Optional[str] = Field(description="Any specific geographical location mentioned.")
    suggested_sources: List[str] = Field(description="A list of 3-5 top-tier, relevant source domains (e.g., 'bbc.co.uk', 'wsj.com').")

class FinalAnswer(BaseModel):
    answer: str = Field(description="The final answer to the user's question, based on internal knowledge.")

class FirstPassOutput(BaseModel):
    output: Union[SearchStrategy, FinalAnswer]

# --- 5. Prompt Definitions ---
DOCUMENT_QA_PROMPT = ChatPromptTemplate.from_template("""(Your original DOCUMENT_QA_PROMPT here)""")
RESEARCH_AGENT_PROMPT = ChatPromptTemplate.from_template("""(Your original RESEARCH_AGENT_PROMPT here)""")

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""You are an expert search strategist. Your job is to analyze the user's query and decide if a search is needed. If it is, you must devise a search strategy.

**Analyze the USER'S QUESTION:**
1.  **Search Requirement:** Is the question about real-time events, news, specific dates, or niche topics that require up-to-date information?
2.  **Topic Extraction:** What is the core subject of the query?
3.  **Location Identification:** Is a specific country, city, or region mentioned?
4.  **Source Suggestion:** Based on the topic and location, what are the 3-5 most authoritative and relevant websites to search? (e.g., for UK news, suggest 'bbc.co.uk', 'theguardian.com'; for US finance, suggest 'bloomberg.com', 'wsj.com').

You must respond in ONE of the following two JSON formats:

1.  If no search is needed (general knowledge):
    `{{"output": {{"answer": "Your detailed and helpful answer here."}}}}`

2.  If a search is required:
    `{{"output": {{"requires_search": true, "topic": "...", "location": "...", "suggested_sources": ["...", "..."]}}}}`

USER'S QUESTION: {question}
CHAT HISTORY: {chat_history}""")

EXPERT_QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_template("""You are a search query generator. Take the provided search strategy and the original user question to create a single, expert-level search query string for a search engine.

**Instructions:**
- Combine the topic and location into a concise query.
- Use the `source:...` operator to prioritize the suggested sources. Combine them with `OR`.

**Search Strategy:**
{strategy}

**Original User Question:**
{original_question}

Return only the final search query string.
""")

# --- 6. Core Helper and Agent Functions ---
# (get_llm_for_user, format_docs, key helpers, vector store, scraper functions all remain the same)
def get_llm_for_user(model_key: str): # Stub
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    return ChatDeepInfra(model=model_id, temperature=0.7)
def format_docs(docs: List[Document]) -> str: # Stub
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        if "source" in doc.metadata:
            content += f" (Source: {doc.metadata['source']})"
        formatted_docs.append(content)
    return "\n\n".join(formatted_docs)
def get_user_active_collection_key(user_id: str) -> str: return f"user:{user_id}:active_collection"
def get_user_active_filename_key(user_id: str) -> str: return f"user:{user_id}:active_filename"
def create_and_store_vector_store(...): pass # Stub
def manual_retriever(...): pass # Stub
async def scrape_url(...): pass # Stub
async def scrape_and_filter_urls(...): pass # Stub
async def run_tavily_research_agent(query: str) -> List[Document]:
    logging.info(f"--- Running Tavily Research Agent with expert query: '{query}' ---")
    if not TAVILY_API_KEY:
        logging.error("Tavily API key not set. Search will fail.")
        return []
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        search_results = await asyncio.to_thread(tavily_client.search, query=query, search_depth="advanced", max_results=7)
        unique_urls = [item['url'] for item in search_results.get('results', [])]
        logging.info(f"Tavily Agent found {len(unique_urls)} URLs.")
        return await scrape_and_filter_urls(unique_urls)
    except Exception as e:
        logging.error(f"Tavily search failed: {e}", exc_info=True)
        return []

# --- 7. FastAPI Endpoints ---
class ChatRequest(BaseModel):
    conversation_id: str
    model_key: str
    message: str

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...), current_user: User = Security(get_current_user)):
    # (implementation remains)
    pass

@app.post("/start_new_chat")
def start_new_chat(current_user: User = Security(get_current_user)):
    # (implementation remains)
    pass

@app.get("/get_active_document")
def get_active_document(current_user: User = Security(get_current_user)):
    # (implementation remains)
    pass


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
        # (Document RAG logic remains the same)
        mode = "DOCUMENT_QA"
        pass
    else:
        # ✅ NEW: Dynamic Two-Step Strategy Agent
        strategy_parser = JsonOutputParser(pydantic_object=FirstPassOutput)
        strategy_chain = QUERY_ANALYSIS_PROMPT | synthesis_llm | strategy_parser
        
        logging.info("--- Step 1: Analyzing Query for Strategy ---")
        strategy_result = await strategy_chain.ainvoke({
            "question": request.message,
            "chat_history": history_string
        })
        
        decision = strategy_result['output']
        
        if 'requires_search' in decision and decision['requires_search']:
            logging.info(f"Generated search strategy: {decision}")
            mode = "RESEARCH_AGENT (Dynamic Strategy)"

            # Step 2: Generate the expert query
            query_generator_chain = EXPERT_QUERY_GENERATOR_PROMPT | synthesis_llm | StrOutputParser()
            expert_query = await query_generator_chain.ainvoke({
                "strategy": json.dumps(decision),
                "original_question": request.message
            })
            
            documents = await run_tavily_research_agent(expert_query)
            if not documents:
                response_text = "I tried to search for that, but I couldn't find enough information online."
            else:
                context_string = format_docs(documents)
                synthesis_chain = RESEARCH_AGENT_PROMPT | synthesis_llm | StrOutputParser()
                response_text = await synthesis_chain.ainvoke({"context": context_string, "question": request.message})
        
        elif 'answer' in decision:
            logging.info("Strategy analysis answered directly from knowledge.")
            mode = "PURE_CHAT (Confident Answer)"
            response_text = decision['answer']
        
        else:
            logging.error(f"Unexpected output from strategy analysis: {decision}")
            response_text = "I encountered an error planning how to answer your question."
            mode = "ERROR"

    # Save history and return response
    try:
        history_to_save = [{"type": "human", "content": request.message}, {"type": "ai", "content": response_text}]
        for message in history_to_save:
            REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps(message))
        
        final_model_used = synthesis_llm.model_name
        return {
            "response": response_text, 
            "model_used": final_model_used, 
            "conversation_id": request.conversation_id, 
            "mode": mode
        }
    except Exception as e:
        logging.error(f"Final response stage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 8. Utility Endpoints ---
@app.get("/get_conversations")
def get_conversations(current_user: User = Security(get_current_user)):
    # (implementation remains)
    pass

@app.get("/history/{conversation_id}")
def get_history(conversation_id: str, current_user: User = Security(get_current_user)):
    # (implementation remains)
    pass