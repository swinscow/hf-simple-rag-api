# main.py - Consolidated Code for Deep Infra RAG PoC

# --- 1. Dependencies and Setup ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv # Loads .env file
import os                     # Used to read environment variables
import shutil                 # Used for file operations (saving upload)

# LangChain Components for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepinfra import ChatDeepInfra # Deep Infra LLM Connector
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# --- 2. Configuration and Initialization ---
# Load environment variables from .env file
load_dotenv()

# --- CRITICAL FIX: Set the variable the library is looking for ---
deepinfra_key = os.getenv("DEEPINFRA_API_KEY") 

if not deepinfra_key:
    raise ValueError("DEEPINFRA_API_KEY not found. Check your .env file!")
else:
    # 1. Set the variable the underlying library will check
    os.environ['OPENAI_API_KEY'] = deepinfra_key
    # 2. Set the custom base URL (which you already had, now slightly cleaner)
    os.environ['OPENAI_API_BASE'] = "https://api.deepinfra.com/v1/openai" 
    
    print(f"SUCCESS: Key loaded and environment configured for Deep Infra. Key starts with: {deepinfra_key[:10]}")

app = FastAPI()

# Configuration Variables
CHROMA_DB_DIR = "./local_chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # The open-source model Deep Infra will run
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# CRITICAL: Retrieve the Deep Infra key from the environment
deepinfra_key = os.getenv("DEEPINFRA_API_KEY") 

if not deepinfra_key:
    # This check prevents the authentication error if the .env file is bad
    raise ValueError("DEEPINFRA_API_KEY not found. Please check your .env file.")

# Initialize the Deep Infra LLM client (connects to the external GPU)
# We pass 'openai_api_key' and 'openai_api_base' for compatibility with the API wrapper.
llm = ChatDeepInfra(
    openai_api_key=deepinfra_key,
    model=LLM_MODEL,
    temperature=0.7,
)

# Global variable to hold the vector store instance
vector_store = None 

# --- 3. RAG Pipeline Functions ---

def create_vector_store(file_path: str):
    """Loads, chunks, embeds, and indexes the document into a local ChromaDB."""
    try:
        # 1. Load data (Requires pypdf)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 2. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 3. Create embeddings locally (High Privacy: uses local CPU/RAM)
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

        # 4. Create and persist the local vector store (ChromaDB)
        global vector_store
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_DIR
        )
        vector_store.persist()
        return True
    except Exception as e:
        print(f"RAG creation failed: {e}")
        return False

# Standard RAG Prompt Template - for instructing the LLM
RAG_PROMPT = """
You are an expert Q&A assistant. Answer the user's question only based on the provided context.
If the context does not contain the answer, state that you cannot find the information in the documents.

CONTEXT:
{context}

QUESTION:
{question}
"""
prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


# --- 4. FastAPI Endpoints ---

# Pydantic model for the chat request body
class ChatRequest(BaseModel):
    message: str


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a file, saves it, and creates the local RAG index."""
    # Ensure a temp_files directory exists
    os.makedirs("./temp_files", exist_ok=True)
    
    # Create the file path
    file_path = os.path.join("./temp_files", file.filename)
    
    # Save the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file and create the RAG index
    if create_vector_store(file_path):
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully!"}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")


@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Retrieves context from local documents and generates a response via the LLM API."""
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document index loaded. Please upload a file first.")

    # 1. Retrieval: Search the local index for relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieves top 3 chunks

    # 2. RAG Chain: Defines the steps: retrieve -> augment prompt -> generate
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 3. Generation & Return
    try:
        # Pass the user message through the RAG chain
        response = rag_chain.invoke(request.message)
        return {"response": response}
    except Exception as e:
        # Catches API key errors, rate limits, etc.
        # CRITICAL: If you get an error here, the key/model/URL is likely wrong.
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")