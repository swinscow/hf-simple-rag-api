# main.py - FINAL CORRECTED VERSION (with Intelligent Query Router)

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
DOCUMENT_QA_PROMPT = ChatPromptTemplate.from_template("""You are an expert Q&A assistant for user-provided documents. Your goal is to answer the user's QUESTION using the information from the DOCUMENT CONTEXT below. If the context is insufficient, you may use your general knowledge. IMPORTANT: Do not refer to the document or the context in your response. Answer the question directly.

DOCUMENT CONTEXT:
{context}
CHAT HISTORY:
{chat_history}
QUESTION:
{question}
""")

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

ROUTER_PROMPT = ChatPromptTemplate.from_template("""You are a hyper-efficient query routing assistant. Your sole job is to analyze the user's LATEST QUESTION and determine if a web search is absolutely necessary to answer it.

**CRITICAL RULES:**
- A web search is **REQUIRED** if the question involves:
  - Any recent or future dates (e.g., "yesterday," "next week," "in October 2025").
  - Real-time information (e.g., "what is the stock price of...?", "latest news," "weather today").
  - Very specific or niche topics, people, or products that are not common knowledge.
- A web search is **NOT** required for:
  - General knowledge, historical facts, definitions (e.g., "Who was Shakespeare?", "What is gravity?").
  - Simple conversational follow-ups (e.g., "Thank you," "Tell me more," "Can you rephrase that?").

**EXAMPLES:**
- User Question: "What were the main UK news headlines on October 15th, 2025?" -> **Classification: RESEARCH_REQUIRED** (Reason: Specific future date requires real-time news lookup).
- User Question: "Who was the first person on the moon?" -> **Classification: GENERAL_KNOWLEDGE** (Reason: A well-established historical fact).
- User Question: "That's interesting, thanks!" -> **Classification: CONVERSATIONAL** (Reason: Simple conversational response).

---
**CONTEXT:**
CHAT HISTORY:
{chat_history}

LATEST QUESTION:
{question}

---
Respond with a JSON object containing your classification based on the rules and examples above.
Example: {{"query_type": "RESEARCH_REQUIRED"}}""")

# --- 5. Core Helper and Agent Functions ---
# ✅ CORRECTED: All helper functions are now fully included.
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
        active_filename_key = get_user_active_filename_key(user_id)
        REDIS_CLIENT_INSTANCE.set(active_filename_key, original_filename)
        return True
    except Exception as e:
        logging.error(f"RAG creation failed for {collection_name}: {e}", exc_info=True)
        return False

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

async def scrape_url(url: str) -> Optional[Document]:
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = ' '.join(p.get_text() for p in soup.find_all('p'))
        if len(text_content) > 100:
            return Document(page_content=text_content, metadata={"source": url})
        return None
    except Exception as e:
        logging.warning(f"Failed to scrape {url}: {e}")
        return None

async def scrape_and_filter_urls(urls: List[str]) -> List[Document]:
    scrape_tasks = [scrape_url(url) for url in urls]
    scraped_documents = await asyncio.gather(*scrape_tasks)
    final_documents = [doc for doc in scraped_documents if doc]
    logging.info(f"Successfully scraped {len(final_documents)}/{len(urls)} URLs.")
    return final_documents

async def run_tavily_research_agent(query: str) -> List[Document]:
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

class RouterResponse(BaseModel):
    query_type: Literal["RESEARCH_REQUIRED", "GENERAL_KNOWLEDGE", "CONVERSATIONAL"] = Field(
        description="The classification of the user's query."
    )

async def route_query(question: str, chat_history: str) -> str:
    logging.info(f"Routing query: {question}")
    router_llm = get_llm_for_user("fast-chat")
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
    active_collection_key = get_user_active_collection_key(current_user.user_id)
    active_filename_key = get_user_active_filename_key(current_user.user_id)
    REDIS_CLIENT_INSTANCE.delete(active_collection_key, active_filename_key)
    return {"message": "New chat session started, document context cleared."}

@app.get("/get_active_document")
def get_active_document(current_user: User = Security(get_current_user)):
    active_filename_key = get_user_active_filename_key(current_user.user_id)
    filename = REDIS_CLIENT_INSTANCE.get(active_filename_key)
    if filename:
        return {"active_filename": filename.decode('utf-8')}
    return {"active_filename": None}

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
        collection_name_str = active_collection_name.decode('utf-8')
        logging.info(f"User {user_id} using RAG with collection: {collection_name_str}")
        mode = "DOCUMENT_QA"
        retriever = RunnableLambda(manual_retriever)
        chain_input = {"collection_name": collection_name_str, "question": request.message, "chat_history": history_string}
        chat_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
            | DOCUMENT_QA_PROMPT | llm_instance | StrOutputParser()
        )
        response_text = await chat_chain.ainvoke(chain_input)
    elif request.use_search:
        query_type = await route_query(request.message, history_string)
        
        if query_type == "RESEARCH_REQUIRED":
            mode = "RESEARCH_AGENT (Tavily)"
            documents = await run_tavily_research_agent(request.message)
            if not documents:
                response_text = "I tried to search for that, but I couldn't find enough information online."
            else:
                context_string = format_docs(documents)
                synthesis_chain = RESEARCH_AGENT_PROMPT | synthesis_llm | StrOutputParser()
                response_text = await synthesis_chain.ainvoke({"context": context_string, "question": request.message})
        else:
            mode = f"PURE_CHAT ({query_type})"
            chat_chain = PURE_CHAT_PROMPT | llm_instance | StrOutputParser()
            response_text = await chat_chain.ainvoke({"question": request.message, "chat_history": history_string})
    else:
        mode = "PURE_CHAT (Search Disabled)"
        chat_chain = PURE_CHAT_PROMPT | llm_instance | StrOutputParser()
        response_text = await chat_chain.ainvoke({"question": request.message, "chat_history": history_string})

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