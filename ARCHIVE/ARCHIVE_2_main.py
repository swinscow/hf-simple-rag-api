# main.py - FINAL CONSOLIDATED CODE FOR CLOUD MIGRATION FOUNDATION

# --- 1. Dependencies and Setup ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv # Loads .env file
import os                     # Used to read environment variables
import shutil                 # Used for file operations (saving upload)
from typing import Dict, List, Any
from operator import itemgetter 
import uuid                   # For generating unique IDs

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
# The operator module is built-in, but itemgetter is explicitly imported for clarity.

# --- Pydantic Model for Intent Classification ---
class GroundingDecision(BaseModel):
    """Model to force the LLM to output a structured decision."""
    is_general_knowledge: bool = Field(
        ..., 
        description="True if the user is asking a question that requires external, non-document knowledge, OR if the user explicitly tells the model to use its general knowledge. False if the user asks to stick strictly to the document."
    )

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
# Key: User ID (str) -> Value: Dictionary of {Conversation ID (str) -> List of Messages (List[Any])}
CONVERSATION_HISTORY: Dict[str, Dict[str, List[Any]]] = {} 

# Global variable for RAG index
vector_store = None 


# --- 3. Prompt Definitions ---
# 1. STRICT RAG PROMPT (For Smart RAG = False / Strict Mode)
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


# 2. FLEXIBLE RAG PROMPT (For Smart RAG = True / General Knowledge Mode)
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

# 3. PURE CHAT PROMPT (Used when no document is uploaded)
PURE_CHAT_PROMPT = """
You are a friendly and helpful general knowledge assistant. Use your knowledge to answer the user's question and maintain the conversation flow.

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""
pure_prompt = ChatPromptTemplate.from_template(PURE_CHAT_PROMPT)


# --- 4. Core Helper Functions ---

def get_llm_for_user(model_key: str):
    """Dynamically creates the LLM instance based on user's model choice."""
    model_id = MODEL_MAPPING.get(model_key, MODEL_MAPPING["fast-chat"])
    return ChatDeepInfra(
        model=model_id,
        temperature=0.7,
    )

def check_for_general_intent(message: str, llm_instance: ChatDeepInfra) -> bool:
    """Uses the LLM to classify whether the user intends to use general knowledge."""
    # We use a very strict classification prompt
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
        # We want STRICT RAG if is_general_knowledge is False (meaning user said 'stick to the file')
        return not decision.is_general_knowledge
    except Exception as e:
        # Fallback to the safer method if the LLM classification fails
        print(f"Warning: LLM classification failed ({e}). Falling back to keyword search.")
        return any(keyword in message.lower() for keyword in ["only use the file", "only based on the file", "only use the document", "strictly only"])

def create_vector_store(file_path: str):
    """Loads, chunks, embeds, and indexes the document into a local ChromaDB."""
    try:
        # Load data (handles PDF, requires pypdf)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create embeddings locally (High Privacy: uses local CPU/RAM)
        # NOTE: Using the global EMBEDDINGS_INSTANCE defined in the setup section
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

def format_docs(docs):
    """Converts a list of documents into a single string for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 5. FastAPI Endpoints ---

# Updated Pydantic model for request body
class ChatRequest(BaseModel):
    user_id: str = "default_tester"
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


@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """Handles chat, RAG, memory, and dynamic grounding based on user message."""
    global vector_store

    # 1. Initialization and History Retrieval
    CONVERSATION_HISTORY.setdefault(request.user_id, {})
    llm_instance = get_llm_for_user(request.model_key)
    
    history = CONVERSATION_HISTORY[request.user_id].get(request.conversation_id, [])
    history_string = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history])
    
    # 2. Logic Branching
    if vector_store is not None:
        # --- RAG MODE (Document Available) ---
        
        # Determine strictness by checking LLM's intent classification
        is_strict = check_for_general_intent(request.message, llm_instance)
        
        selected_prompt = strict_prompt if is_strict else flexible_prompt
        mode = "STRICT_RAG" if is_strict else "FLEXIBLE_RAG"
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # RAG Chain: Retrieves documents, formats, and uses the selected prompt
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
        
        # Pure Chat Chain: Runs the general knowledge prompt (no retriever involved)
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
        new_history = history + [
            HumanMessage(content=request.message),
            AIMessage(content=response_text)
        ]
        CONVERSATION_HISTORY.setdefault(request.user_id, {})[request.conversation_id] = new_history
        
        return {"response": response_text, "model_used": llm_instance.model_name, "conversation_id": request.conversation_id, "mode": mode}
        
    except Exception as e:
        print(f"LLM Inference Detail Error: {e}") 
        raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")


# --- 6. Utility Endpoints ---

@app.get("/get_conversations/{user_id}")
def get_conversations(user_id: str):
    """Returns a list of all saved conversation IDs and a snippet of the first message."""
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


@app.get("/history/{user_id}")
def get_history(user_id: str):
    """Debug endpoint to view the current in-memory chat history."""
    return {"history": CONVERSATION_HISTORY.get(user_id, {})}