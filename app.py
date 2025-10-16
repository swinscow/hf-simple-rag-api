# app.py - Streamlit Frontend with Search Toggle

import streamlit as st
import requests
import json
import uuid
from supabase import create_client, Client

# --- CONFIGURATION ---
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
st.title("Welcome to Pueblo. Globally-affordable AI that puts the user first. (MVP)")

# --- Session State Management ---
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'use_search' not in st.session_state:
    st.session_state.use_search = False # NEW: Initialize search state

# --- Authentication Logic ---
def show_login_form():
    # ... (code is unchanged)
    st.header("Login / Sign Up")
    # ...

# --- API Call Functions ---
def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}

def upload_and_index_document(file_to_upload):
    # ... (code is unchanged)
    with st.spinner("Uploading and indexing document..."):
        # ...
        st.success("Document indexed!")

def clear_document():
    # ... (code is unchanged)
    with st.spinner("Clearing document context..."):
        # ...
        st.success("Context cleared!")

def call_api_and_get_response(prompt_text):
    headers = {"Content-Type": "application/json", **get_auth_headers()}
    payload = {
        "conversation_id": st.session_state.conversation_id,
        "model_key": st.session_state.model_key,
        "message": prompt_text,
        "use_search": st.session_state.use_search, # NEW: Pass the search setting
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
    # ... (code is unchanged)
    return []

def load_conversation_history(conversation_id):
    # ... (code is unchanged)
    st.rerun()

# --- Main App Logic ---
if not st.session_state.auth_token:
    show_login_form()
else:
    with st.sidebar:
        st.header("Settings")
        st.session_state.model_key = st.selectbox("Select Model", options=["fast-chat", "smart-chat", "coding-expert"])
        
        # NEW: Toggle switch for internet search
        st.session_state.use_search = st.toggle("Enable Internet Search", value=st.session_state.use_search)
        st.caption("Note: Document chat takes priority over search.")
        
        st.markdown("---")
        if st.button("➕ Start New Chat", use_container_width=True):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        st.caption(f"Session: {st.session_state.conversation_id[:8]}...")
        st.markdown("---")
        st.subheader("Chat History")
        # ... (history display logic is unchanged)
        st.markdown("---")
        st.subheader("RAG Document")
        uploaded_file = st.file_uploader("Upload PDF to chat with", type="pdf")
        if uploaded_file:
            if st.button("Index Document", use_container_width=True):
                upload_and_index_document(uploaded_file)
        if st.button("Clear Document Context", use_container_width=True):
            clear_document()
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.auth_token = None
            st.session_state.user_id = None
            st.rerun()

    # --- Chat Interface (unchanged) ---
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
