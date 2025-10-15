# app.py - With Multi-User RAG Controls

import streamlit as st
import requests
import json
import uuid
from supabase import create_client, Client

# --- CONFIGURATION (omitted for brevity) ---
API_BASE_URL = "https://fastapi-backend-tq2s.onrender.com/"  # <-- MAKE SURE THIS IS YOUR CORRECT BACKEND URL
SUPABASE_URL = "https://nleucprtizqqofaitqcu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5sZXVjcHJ0aXpxcW9mYWl0cWN1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwNDc3OTcsImV4cCI6MjA3NTYyMzc5N30.3y7GSxNsIcXGSVtcFmdkoR0W12jCOGAYYhkjk6HV4qg" # IMPORTANT: Use the ANON key, not the service key

# ... (Supabase client init) ...

st.set_page_config(layout="wide")
st.title("Welcome to pueblo (MVP version)")

# --- Session State & Login Form (omitted for brevity) ---
# ...

# --- API Call Functions ---
def get_auth_headers(): # ...
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}

# ... (upload_and_index_document, call_api_and_get_response, etc. are the same) ...

# NEW: API function to clear document context
def clear_document():
    """Tells the backend to clear the current RAG document context for the user."""
    with st.spinner("Clearing document context..."):
        try:
            response = requests.post(f"{API_BASE_URL}/clear_document_context", headers=get_auth_headers())
            response.raise_for_status()
            st.success(response.json().get("message", "Context cleared!"))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to clear context: {e}")

# ... (get_all_conversations, load_conversation_history are the same) ...

# --- Main App Logic ---
if not st.session_state.auth_token:
    show_login_form()
else:
    with st.sidebar:
        # ... (Settings, New Chat, History sections are the same) ...

        st.markdown("---")
        st.subheader("RAG Document")
        
        uploaded_file = st.file_uploader("Upload PDF to chat with", type="pdf")
        if uploaded_file:
            if st.button("Index Document", use_container_width=True):
                upload_and_index_document(uploaded_file)
        
        # NEW: Button to clear the RAG context
        if st.button("Clear Document Context", use_container_width=True):
            clear_document()

        st.markdown("---")
        if st.button("Logout", use_container_width=True): # ...
            st.rerun()

    # --- Chat Interface (same as before) ---
    # ...