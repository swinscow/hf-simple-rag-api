# main.py - Consolidated Code with Model Selection and In-Memory History

# --- 1. Dependencies and Setup ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

class GroundingDecision(BaseModel):
    """Model to force the LLM to output a structured decision."""
    is_general_knowledge: bool = Field(
        ..., 
        description="True if the user is asking a question that requires external, non-document knowledge, OR if the user explicitly tells the model to use its general knowledge. False if the user asks to stick strictly to the document."
    )

from dotenv import load_dotenv
import os
import shutil
import json
from typing import Dict, List, Any
import uuid # For generating unique IDs


# LangChain Components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepinfra import ChatDeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter 


# --- 2. Configuration and Initialization ---
load_dotenv()
app = FastAPI()

# --- Model Definitions (User-Facing Name -> Deep Infra ID) ---
MODEL_MAPPING = {
    "fast-chat": "meta-llama/Meta-Llama-3-8B-Instruct",      # Fast and Cheap
    "smart-chat": "meta-llama/Meta-Llama-3-70B-Instruct",    # Slower, More Capable
    "coding-expert": "codellama/CodeLlama-34b-Instruct-hf"   # Example of a specialized model
}

# Configuration Variables
CHROMA_DB_DIR = "./local_chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# CRITICAL: Authentication and Environment Check
deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
if not deepinfra_key:
    raise ValueError("DEEPINFRA_API_KEY not found. Check your .env file!")
os.environ['OPENAI_API_KEY'] = deepinfra_key 
os.environ['OPENAI_API_BASE'] = DEEPINFRA_BASE_URL

# Temporary In-Memory History Store (Simulates your database)
# Key: User/Session ID (str) -> Value: List of [HumanMessage, AIMessage]
CONVERSATION_HISTORY: Dict[str, Dict[str, List[Any]]] = {} 

# Global variable for RAG index
vector_store = None 

# main.py - Add this Intent Classification function

def check_for_general_intent(message: str, llm_instance: ChatDeepInfra) -> bool:
    """Uses the LLM to classify whether the user intends to use general knowledge."""
    
    # We use a very strict classification prompt
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an Intent Classifier. Your task is to determine if the user intends "
         "to use general knowledge (True) or strictly restrict the answer to a document (False). "
         "Output ONLY the JSON object. Default to True unless the message strictly implies restriction."
         "Example User Inputs: 'Only use the file', 'Strictly answer from the document', 'Stick to the context'."
        ),
        ("user", "User Message: {message}")
    ])
    
    classification_chain = (
        classification_prompt
        | llm_instance.with_structured_output(GroundingDecision) # <-- Forces JSON output
    )

    try:
        decision = classification_chain.invoke({"message": message})
        # If the LLM says it's NOT general knowledge (i.e., strict), we flip the boolean for our strict flag
        return not decision.is_general_knowledge
    except Exception as e:
        # Fallback to the original safe keyword detection if the classification fails
        print(f"Warning: LLM classification failed ({e}). Falling back to keyword search.")
        return False # Default to flexible RAG mode if LLM fails

# --- TEMPORARY STARTUP FIX ---
# Delete the line above: 'vector_store = None'
# And replace it with this:
try:
    # We can safely load the embeddings model once at startup without crashing
    EMBEDDINGS_INSTANCE = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e:
    print(f"FATAL: Embedding Model failed to load: {e}")
    EMBEDDINGS_INSTANCE = None # Set to None so the app still starts

vector_store = None 
# -----------------------------

# --- 3. RAG and LLM Core Functions ---

# Function to dynamically create the LLM instance based on user choice
def get_llm_for_user(model_key: str):
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    
    # Initialize the Deep Infra LLM client using environment variables
    # The API key and Base URL are read from os.environ
    return ChatDeepInfra(
        model=model_id,
        temperature=0.7,
    )

def create_vector_store(file_path: str):
    # ... (Keep the original PyPDFLoader, Chunking, and ChromaDB logic here)
    # NOTE: You must still manually paste the code from the previous working step into this function!
    # [Start of previous working code block for create_vector_store]
    
    try:
        # Load data (handles PDF, install python-docx for .docx support)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create embeddings locally 
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

        # Create and persist the local vector store (ChromaDB)
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
        
    # [End of previous working code block for create_vector_store]

# main.py - Two Prompt Templates

# 1. STRICT RAG PROMPT (For strict_rag=True)
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


# 2. FLEXIBLE RAG PROMPT (For strict_rag=False)
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

# You can now delete the old RAG_PROMPT variable if it was different from STRICT_RAG_PROMPT

# --- Utility to Format Documents ---
def format_docs(docs):
    """Converts a list of documents into a single string for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 4. FastAPI Endpoints ---


# main.py - Updated ChatRequest Pydantic Model

class ChatRequest(BaseModel):
    user_id: str = "default_tester"
    conversation_id: str = "new_chat"
    model_key: str = "fast-chat"
    message: str
    
    # --- NEW CRITICAL FIELD ---
    # If True, the LLM MUST use the RAG context (strict safety). 
    # If False, the LLM can use its general knowledge if RAG context is weak.
    # strict_rag: bool = True

# main.py - Endpoint Restoration (Add this alongside your other endpoints)

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a file, saves it, and creates the local RAG index."""
    # Ensure a temp_files directory exists
    os.makedirs("./temp_files", exist_ok=True)
    
    # Create the file path
    file_path = os.path.join("./temp_files", file.filename)
    
    # Save the uploaded file to disk using shutil
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file and create the RAG index
    if create_vector_store(file_path):
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully!"}
    else:
        # If vector creation failed, throw a server error
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")


# main.py - Updated @app.post("/chat") with Conditional Grounding

# main.py - Updated @app.post("/chat") with Smart RAG Grounding

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Retrieves context, uses conversation history, and generates a response based on mode and grounding choice."""
    global vector_store

    # 1. Initialization and History Retrieval (Same as before)
    # NOTE: user_id and conversation_id are still manually passed for now.
    llm_instance = get_llm_for_user(request.model_key)
    
    # Ensure the user has an entry in the main history dictionary
    CONVERSATION_HISTORY.setdefault(request.user_id, {})
    
    history = CONVERSATION_HISTORY[request.user_id].get(request.conversation_id, [])
    history_string = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history])
    
    # --- CRITICAL CONDITIONAL LOGIC & PROMPT SELECTION ---
    
# --- OLD KEYWORD CHECK ---
    # is_strict = any(keyword in request.message.lower() for keyword in ["only use the file", "only based on the file", "only use the document", "strictly only"])
    # -------------------------

    # --- NEW SMART RAG CHECK ---
    # Determine the strictness using the LLM classifier function
    is_strict = check_for_general_intent(request.message, llm_instance)
    # ---------------------------
    
    if vector_store is not None:
        # --- RAG MODE (Document Available) ---
        mode = "RAG"
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Select the prompt based on LLM's interpretation
        selected_prompt = strict_prompt if is_strict else flexible_prompt
        mode = "STRICT_RAG" if is_strict else "FLEXIBLE_RAG"

        # ... (rest of the RAG chain logic remains the same) ...
            
        # RAG Chain: Retrieves documents, formats, and uses the selected prompt
        chat_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": itemgetter("question") | retriever | format_docs 
            }
            | selected_prompt # <-- Uses the dynamically selected prompt
            | llm_instance
            | StrOutputParser()
        )

    else:
        # --- PURE CHAT MODE (No Document Indexed) ---
        mode = "PURE_CHAT"
        
        # Pure Chat Chain: Runs the general knowledge prompt
        chat_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | pure_prompt
            | llm_instance
            | StrOutputParser()
        )
        
    # 2. Generation & History Update (Remains the same)
    try:
        # ... (History saving logic remains here) ...
        input_dict = {
            "question": request.message,
            "chat_history": history_string,
        }
        
        response_text = chat_chain.invoke(input_dict)
        
        # Update the history store with the new messages (using setdefault for safety)
        new_history = history + [
            HumanMessage(content=request.message),
            AIMessage(content=response_text)
        ]
        CONVERSATION_HISTORY.setdefault(request.user_id, {})[request.conversation_id] = new_history
        
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id, "mode": mode}
        
    except Exception as e:
        print(f"LLM Inference Detail Error: {e}") 
        CONVERSATION_HISTORY.setdefault(request.user_id, {}) # Ensure key exists for history endpoint
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")

@app.get("/get_conversations/{user_id}")
def get_conversations(user_id: str):
    """
    Returns a list of all saved conversation IDs and a snippet of the first message
    for a given user, mimicking a chat history pane.
    """
    user_chats = CONVERSATION_HISTORY.get(user_id, {})
    
    # Format the output for the history pane
    history_pane_data = []
    for conv_id, messages in user_chats.items():
        if messages:
            # The human's first message is typically the title/snippet
            snippet = messages[0].content[:50] + "..." 
            history_pane_data.append({
                "conversation_id": conv_id,
                "snippet": snippet,
                "length": len(messages)
            })
            
    return {"conversations": history_pane_data}

# --- Utility Endpoint for Debugging History (Optional) ---
@app.get("/history/{user_id}")
def get_history(user_id: str):
    """Debug endpoint to view the current in-memory chat history."""
    return {"history": CONVERSATION_HISTORY.get(user_id, [])}






















@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Retrieves context, uses CONVERSATION history, and generates a response."""
    global vector_store
    
    # 1. Validation & Initialization
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document index loaded. Please upload a file first.")
    
    # Ensure the user has an entry in the main history dictionary
    if request.user_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[request.user_id] = {}

    # Get the specific conversation history (returns [] if new chat)
    history = CONVERSATION_HISTORY[request.user_id].get(request.conversation_id, [])

    llm_instance = get_llm_for_user(request.model_key)
    
    # Format history messages for the prompt template
    history_string = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history])
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Retrieval & Augmentation (Same robust chain)
    rag_chain = (
        {
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "context": itemgetter("question") | retriever | format_docs 
        }
        | prompt
        | llm_instance
        | StrOutputParser()
    )

    # 3. Generation & History Update
    try:
        input_dict = {
            "question": request.message,
            "chat_history": history_string,
        }
        
        response_text = rag_chain.invoke(input_dict)
        
        # --- CRITICAL MEMORY WRITE ---
        # Update the history store with the new messages
        new_history = history + [
            HumanMessage(content=request.message),
            AIMessage(content=response_text)
        ]
        
        # Store the updated list back into the nested dictionary
        CONVERSATION_HISTORY[request.user_id][request.conversation_id] = new_history
        
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id}
        
    except Exception as e:
        print(f"LLM Inference Detail Error: {e}") 
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")




















# Updated Pydantic model to include session ID and model choice
class ChatRequest(BaseModel):
    user_id: str = "default_tester"  # Simulates a logged-in user
    model_key: str = "fast-chat"      # User's model choice
    message: str


@app.post("/upload_document")
# This endpoint remains largely the same
async def upload_document(file: UploadFile = File(...)):
    """Accepts a file, saves it, and creates the local RAG index."""
    os.makedirs("./temp_files", exist_ok=True)
    file_path = os.path.join("./temp_files", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if create_vector_store(file_path):
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully!"}
    else:
        raise HTTPException(status_code=500, detail="Failed to create RAG index.")


@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Retrieves context, uses conversation history, and generates a response."""
    global vector_store
    
    # 1. Validation & Initialization
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document index loaded. Please upload a file first.")
    
    # Get the correct LLM based on user's choice
    llm_instance = get_llm_for_user(request.model_key)
    
    # Get conversation history for this user/session
    history = CONVERSATION_HISTORY.get(request.user_id, [])
    history_string = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history])
    
    # 2. Retrieval & Augmentation
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    setup_and_retrieval = (
    {
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        "context": itemgetter("question") | retriever, # Retriever outputs documents here
    }
)
    # Combine RAG context and Chat History into the prompt variables
    rag_chain = (
        setup_and_retrieval
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"])) # <-- CRITICAL FIX: Formats docs into a single string
        | prompt
        | llm_instance
        | StrOutputParser()
)

    # 3. Generation & History Update
    try:
        # Construct the input dictionary with all three variables
        input_dict = {
            "question": request.message,
            "chat_history": history_string,
        }
        
        # Invoke the chain
        response_text = rag_chain.invoke(input_dict)
        
        # Update the in-memory history (Crucial simulation of database write)
        CONVERSATION_HISTORY.setdefault(request.user_id, []).append(HumanMessage(content=request.message))
        CONVERSATION_HISTORY[request.user_id].append(AIMessage(content=response_text))
        
        return {"response": response_text, "model_used": llm_instance.model_name}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")

# --- Utility Endpoint for Debugging History (Optional) ---
@app.get("/history/{user_id}")
def get_history(user_id: str):
    """Debug endpoint to view the current in-memory chat history."""
    return {"history": CONVERSATION_HISTORY.get(user_id, [])}