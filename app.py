# app.py - Streamlit Frontend Client

import streamlit as st
import requests
import json
import uuid
from datetime import datetime

# --- CONFIGURATION (UPDATE THIS FOR DEPLOYMENT) ---
# Use your Render public URL as the base API endpoint
API_BASE_URL = "https://hf-simple-rag-api.onrender.com"  # <-- REPLACE WITH YOUR ACTUAL RENDER URL
TEST_USER_ID = "SUPABASE_REDIS_TESTER" # Same ID used for the current memory storage
# ---------------------------------------------------

st.set_page_config(layout="wide")
st.title("🧠 Open-Source LLM Platform (MVP)")

# --- Sidebar: Model and Session Management ---

# Get list of model keys for the dropdown
MODEL_KEYS = ["fast-chat", "smart-chat", "coding-expert"]

# Use Streamlit session state to manage user ID and current conversation ID
if 'user_id' not in st.session_state:
    # In a real app, this would be retrieved from a login token (Auth0/Supabase)
    st.session_state.user_id = TEST_USER_ID
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # Model Selection Dropdown
    selected_model_key = st.selectbox(
        "Select Model",
        options=MODEL_KEYS,
        index=MODEL_KEYS.index("fast-chat") 
    )
    st.session_state.model_key = selected_model_key
    
    st.markdown("---")
    
    # Button to start a fresh chat
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.experimental_rerun()
        
    st.caption(f"Current Session: {st.session_state.conversation_id[:8]}...")
    st.caption(f"User ID: {st.session_state.user_id}")

    st.markdown("---")
    
    # Placeholder for the Document Upload Feature (Simple Text)
    st.subheader("RAG Indexing")
    st.info("The document upload endpoint is active on the backend.")
    # In a fully integrated app, the upload component would be here.

# --- Main Chat Interface ---

def call_api_and_get_response(prompt_text):
    """Sends the user prompt to the live FastAPI backend."""
    headers = {
        "Content-Type": "application/json",
        # In a real app, the secure token would be passed here:
        # "Authorization": f"Bearer {st.session_state.auth_token}" 
    }
    
    # 1. Build the payload matching the ChatRequest Pydantic model
    payload = {
        "conversation_id": st.session_state.conversation_id,
        "model_key": st.session_state.model_key,
        "message": prompt_text,
    }
    
    # 2. Call the deployed API
    try:
        response = requests.post(f"{API_BASE_URL}/chat", headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        return data.get("response", "Error: LLM returned no text."), data.get("mode", "UNKNOWN")
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Failed to connect or receive response. Is the Render service spinning up? ({e})")
        return "Sorry, the server is currently unavailable or timed out.", "ERROR"


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question or provide instructions..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API and stream the response
    with st.spinner("Thinking..."):
        ai_response, mode = call_api_and_get_response(prompt)

    # Append AI response to history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        st.caption(f"Mode: {mode} | Model: {st.session_state.model_key}")