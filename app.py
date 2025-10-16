# app.py - FINAL STABLE VERSION

import streamlit as st
import requests
import json
import uuid
from supabase import create_client, Client

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual URLs and Keys
API_BASE_URL = "https://fastapi-backend-tq2s.onrender.com"  # <-- MAKE SURE THIS IS YOUR CORRECT BACKEND URL
SUPABASE_URL = "https://nleucprtizqqofaitqcu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5sZXVjcHJ0aXpxcW9mYWl0cWN1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwNDc3OTcsImV4cCI6MjA3NTYyMzc5N30.3y7GSxNsIcXGSVtcFmdkoR0W12jCOGAYYhkjk6HV4qg" # IMPORTANT: Use the ANON key, not the service key


# --- INITIALIZE SUPABASE CLIENT ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Could not connect to Supabase. Check your URL and Key. Error: {e}")
    st.stop()

st.set_page_config(layout="wide")
st.title("🧠 Open-Source LLM Platform")

# --- SESSION STATE INITIALIZATION ---
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'use_search' not in st.session_state:
    st.session_state.use_search = False
if 'use_premium_search' not in st.session_state: # NEW: Added premium search state
    st.session_state.use_premium_search = False
if 'active_document' not in st.session_state:
    st.session_state.active_document = None


# --- AUTHENTICATION LOGIC ---
def show_login_form():
    st.header("Login / Sign Up")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Login", use_container_width=True):
                try:
                    auth_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.auth_token = auth_response.session.access_token
                    st.session_state.user_id = auth_response.user.id
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
        with col2:
            if st.form_submit_button("Sign Up", use_container_width=True):
                try:
                    supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Sign up successful! Please check your email to verify.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")


# --- API CALL FUNCTIONS ---
def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}

def check_active_document():
    """Asks the backend if a document is currently active for the user."""
    try:
        response = requests.get(f"{API_BASE_URL}/get_active_document", headers=get_auth_headers())
        response.raise_for_status()
        data = response.json()
        st.session_state.active_document = data.get("active_filename")
    except requests.exceptions.RequestException:
        st.session_state.active_document = None


def upload_and_index_document(file_to_upload):
    with st.spinner("Uploading and indexing document..."):
        files = {"file": (file_to_upload.name, file_to_upload, "application/pdf")}
        try:
            response = requests.post(f"{API_BASE_URL}/upload_document", headers=get_auth_headers(), files=files)
            response.raise_for_status()
            st.success(response.json().get("message", "Document indexed successfully!"))
            st.session_state.active_document = file_to_upload.name
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to upload document: {e}")


def start_new_chat_session():
    """Calls the backend to ensure all context is cleared."""
    with st.spinner("Starting new chat..."):
        try:
            response = requests.post(f"{API_BASE_URL}/start_new_chat", headers=get_auth_headers())
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Failed to start new session: {e}")


def call_api_and_get_response(prompt_text):
    headers = {"Content-Type": "application/json", **get_auth_headers()}
    # MODIFIED: Added the new 'use_premium_search' flag to the payload
    payload = {
        "conversation_id": st.session_state.conversation_id,
        "model_key": st.session_state.model_key,
        "message": prompt_text,
        "use_search": st.session_state.use_search,
        "use_premium_search": st.session_state.use_premium_search
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Error: No text in response."), data.get("mode", "UNKNOWN")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return "Server connection failed.", "ERROR"

def get_all_conversations():
    try:
        response = requests.get(f"{API_BASE_URL}/get_conversations", headers=get_auth_headers())
        response.raise_for_status()
        return response.json().get("conversation_ids", [])
    except requests.exceptions.RequestException:
        st.sidebar.error("Could not load history.")
        return []

def load_conversation_history(conversation_id):
    try:
        start_new_chat_session()
        response = requests.get(f"{API_BASE_URL}/history/{conversation_id}", headers=get_auth_headers())
        response.raise_for_status()
        history = response.json().get("history", [])
        st.session_state.messages = []
        for msg in history:
            role = "user" if msg.get("type") == "human" else "assistant"
            st.session_state.messages.append({"role": role, "content": msg.get("content")})
        st.session_state.conversation_id = conversation_id
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load conversation: {e}")


# --- MAIN APP LOGIC ---
if not st.session_state.auth_token:
    show_login_form()
else:
    if st.session_state.active_document is None:
        check_active_document()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Settings")
        st.session_state.model_key = st.selectbox("Select Model", options=["fast-chat", "smart-chat", "coding-expert"])
        
        # MODIFIED: Added logic for the new premium search toggle
        is_rag_active = st.session_state.active_document is not None
        st.session_state.use_search = st.toggle("Enable Internet Search", value=False, disabled=is_rag_active)
        
        if st.session_state.use_search:
            st.session_state.use_premium_search = st.toggle("✨ Premium Search (Tavily)", value=False, help="Uses a more advanced search provider for higher quality results at a higher cost.")

        st.markdown("---")
        
        if st.button("➕ Start New Chat", use_container_width=True):
            start_new_chat_session()
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.active_document = None
            st.rerun()
            
        st.caption(f"Session: {st.session_state.conversation_id[:8]}...")
        st.markdown("---")
        
        st.subheader("Chat History")
        conversations = get_all_conversations()
        if not conversations:
            st.caption("No past conversations found.")
        else:
            for conv_id in conversations:
                if st.button(f"Chat: {conv_id[:8]}...", key=conv_id, use_container_width=True):
                    load_conversation_history(conv_id)
        
        st.markdown("---")
        st.subheader("RAG Document")
        if st.session_state.active_document:
            st.info(f"Active document: **{st.session_state.active_document}**")
            st.caption("Click 'Start New Chat' to clear the document.")
        else:
            uploaded_file = st.file_uploader("Upload PDF to chat with", type="pdf")
            if uploaded_file:
                if st.button("Index Document", use_container_width=True):
                    upload_and_index_document(uploaded_file)
        
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.auth_token = None
            st.session_state.user_id = None
            st.session_state.active_document = None
            st.rerun()

    # --- CHAT INTERFACE ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            ai_response, mode = call_api_and_get_response(prompt)
            
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            st.caption(f"Mode: {mode} | Model: {st.session_state.model_key}")