# main.py - FINAL ARCHITECTURE (Multi-Step Research Agent)

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
DOCUMENT_QA_PROMPT = ChatPromptTemplate.from_template("""You are an expert Q&A assistant for user-provided documents. Your goal is to answer the user's QUESTION using the information from the DOCUMENT CONTEXT below. If the context is insufficient, you may use your general knowledge. IMPORTANT: Do not refer to the document or the context in your response. Answer the question directly.

DOCUMENT CONTEXT:
{context}
CHAT HISTORY:
{chat_history}
QUESTION:
{question}
""")

MULTI_STEP_AGENT_PROMPT = """You are a world-class research assistant. Your goal is to answer the user's question with the most accurate, comprehensive, and well-sourced information available.

You will work in a loop of Thought, Action, and Observation.

**1. Thought:** First, analyze the user's query and the conversation history. Break down the problem. Formulate a plan or a hypothesis. Decide if you need to use a tool.
**2. Action:** If you need to gather information, you have one tool available: `search`. You must call this tool with a perfectly optimized, keyword-rich search query.
**3. Observation:** After you use the tool, you will be given the results.

Repeat this process. With each loop, your knowledge will grow. You can perform multiple searches to explore different facets of a topic.

- If you need to search for multiple things, do it in sequence, one search per loop.
- Continue searching until you are confident you have enough information to construct a complete and factual answer.
- Once you have gathered sufficient information, provide a final, comprehensive answer to the user. Do not output "Thought:" or "Action:" in your final response.

You must respond in one of two JSON formats:

A) To use the search tool:
`{{"thought": "Your reasoning here...", "action": {{"tool": "search", "query": "Your expert search query here"}}}`

B) To provide the final answer to the user:
`{{"thought": "I have gathered enough information to answer the user's question.", "final_answer": "Your complete, well-written answer here."}}`

Let's begin.

Current Conversation:
{chat_history}
"""

RESEARCH_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""You are a research analyst. Your task is to synthesize the provided search results into a single, comprehensive answer to the user's original question.

**Critical Instructions:**
1.  **Synthesize, Don't List:** Weave the information together into a single, well-written narrative summary.
2.  **Cite Everything:** **Every single sentence** must be followed by a citation to the source that supports it, like this: [Source: URL].
3.  **No Outside Information:** Do not add any information that is not explicitly found in the provided text.

**SEARCH RESULTS:**
{context}

**USER'S ORIGINAL QUESTION:**
{original_question}
""")


# --- 6. Core Helper and Agent Functions ---
def get_llm_for_user(model_key: str):
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["smart-chat"])
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

async def run_tavily_search(query: str) -> List[Document]:
    logging.info(f"--- Running Tavily Search with query: '{query}' ---")
    if not TAVILY_API_KEY:
        logging.error("Tavily API key not set. Search will fail.")
        return []
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        search_results = await asyncio.to_thread(tavily_client.search, query=query, search_depth="advanced", max_results=5)
        docs = [
            Document(page_content=res.get("content", ""), metadata={"source": res.get("url")})
            for res in search_results.get("results", [])
        ]
        logging.info(f"Tavily Agent found {len(docs)} results.")
        return docs
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
    if REDIS_CLIENT_INSTANCE:
        REDIS_CLIENT_INSTANCE.delete(active_collection_key, active_filename_key)
    return {"message": "New chat session started, document context cleared."}

@app.get("/get_active_document")
def get_active_document(current_user: User = Security(get_current_user)):
    active_filename_key = get_user_active_filename_key(current_user.user_id)
    if REDIS_CLIENT_INSTANCE:
        filename = REDIS_CLIENT_INSTANCE.get(active_filename_key)
        if filename:
            return {"active_filename": filename.decode('utf-8')}
    return {"active_filename": None}

@app.post("/chat")
async def chat_with_rag(request: ChatRequest, current_user: User = Security(get_current_user)):
    user_id = current_user.user_id
    agent_llm = get_llm_for_user(request.model_key)
    
    history_key = f"chat:{user_id}:{request.conversation_id}"
    history_raw = REDIS_CLIENT_INSTANCE.lrange(history_key, 0, -1)
    
    # Format history for the agent
    chat_history_for_agent = ""
    for msg_raw in history_raw:
        msg = json.loads(msg_raw.decode('utf-8'))
        role = "Human" if msg['type'] == 'human' else "Assistant"
        chat_history_for_agent += f"{role}: {msg['content']}\n"
    chat_history_for_agent += f"Human: {request.message}\n"

    active_collection_key = get_user_active_collection_key(user_id)
    active_collection_name = REDIS_CLIENT_INSTANCE.get(active_collection_key)
    
    response_text = ""
    mode = ""

    if active_collection_name:
        collection_name_str = active_collection_name.decode('utf-8')
        logging.info(f"User {user_id} using RAG with collection: {collection_name_str}")
        mode = "DOCUMENT_QA"
        retriever = RunnableLambda(manual_retriever)
        chain_input = {"collection_name": collection_name_str, "question": request.message, "chat_history": chat_history_for_agent}
        chat_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
            | DOCUMENT_QA_PROMPT | agent_llm | StrOutputParser()
        )
        response_text = await chat_chain.ainvoke(chain_input)
    else:
        # Multi-Step Agent Logic
        mode = "MULTI_STEP_AGENT"
        
        all_gathered_docs = []
        
        # Agent loop
        for i in range(3): # Limit to 3 turns to prevent runaways
            logging.info(f"--- Agent Loop, Iteration {i+1} ---")
            
            prompt = ChatPromptTemplate.from_template(MULTI_STEP_AGENT_PROMPT)
            parser = JsonOutputParser()
            agent_chain = prompt | agent_llm | parser

            agent_response = await agent_chain.ainvoke({"chat_history": chat_history_for_agent})

            if "final_answer" in agent_response:
                logging.info("Agent decided to provide final answer.")
                # If we have gathered docs, synthesize them. Otherwise, use the direct answer.
                if all_gathered_docs:
                    synthesis_chain = RESEARCH_SYNTHESIS_PROMPT | agent_llm | StrOutputParser()
                    context_string = format_docs(all_gathered_docs)
                    response_text = await synthesis_chain.ainvoke({
                        "context": context_string,
                        "original_question": request.message
                    })
                else:
                    response_text = agent_response["final_answer"]
                break # Exit the loop
            
            elif "action" in agent_response and agent_response["action"]["tool"] == "search":
                query = agent_response["action"]["query"]
                logging.info(f"Agent action: search(query='{query}')")
                
                search_docs = await run_tavily_search(query)
                all_gathered_docs.extend(search_docs)
                
                observation = "Observation: \n" + "\n\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in search_docs])
                chat_history_for_agent += f"Assistant:\n{json.dumps(agent_response)}\n{observation}\n"
            
            else:
                logging.error(f"Agent produced unexpected output: {agent_response}")
                response_text = "I encountered an error while trying to process your request."
                break
        
        if not response_text:
            logging.warning("Agent loop finished without a final answer. Synthesizing available information.")
            if all_gathered_docs:
                synthesis_chain = RESEARCH_SYNTHESIS_PROMPT | agent_llm | StrOutputParser()
                context_string = format_docs(all_gathered_docs)
                response_text = await synthesis_chain.ainvoke({
                    "context": context_string,
                    "original_question": request.message
                })
            else:
                response_text = "I tried to find information on that, but I was unable to come up with an answer."


    # Save history and return response
    try:
        REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps({"type": "human", "content": request.message}))
        REDIS_CLIENT_INSTANCE.rpush(history_key, json.dumps({"type": "ai", "content": response_text}))
        
        return {
            "response": response_text, 
            "model_used": agent_llm.model_name, 
            "conversation_id": request.conversation_id, 
            "mode": mode
        }
    except Exception as e:
        logging.error(f"Final response stage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 8. Utility Endpoints ---
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