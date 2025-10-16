# app.py - Streamlit Frontend with Supabase Login

import streamlit as st
import requests
import json
import uuid
from supabase import create_client, Client

# --- CONFIGURATION ---
API_BASE_URL = "https://fastapi-backend-tq2s.onrender.com/"  # <-- MAKE SURE THIS IS YOUR CORRECT BACKEND URL

# Use Streamlit secrets for Supabase credentials in a real app
# For now, we'll place them here.
SUPABASE_URL = "https://nleucprtizqqofaitqcu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5sZXVjcHJ0aXpxcW9mYWl0cWN1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwNDc3OTcsImV4cCI6MjA3NTYyMzc5N30.3y7GSxNsIcXGSVtcFmdkoR0W12jCOGAYYhkjk6HV4qg" # IMPORTANT: Use the ANON key, not the service key

# --- INITIALIZE SUPABASE CLIENT ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Could not connect to Supabase. Check your URL and Key. Error: {e}")
    st.stop()


st.set_page_config(layout="wide")
st.title("Welcome to pueblo ai (MVP)")

# --- Session State Management ---
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Authentication Logic ---
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
                    auth_response = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Sign up successful! Please check your email to verify.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")

# --- Main App Logic ---
if not st.session_state.auth_token:
    show_login_form()
else:
    # --- Sidebar: Model and Session Management ---
    MODEL_KEYS = ["fast-chat", "smart-chat", "coding-expert"]
    with st.sidebar:
        st.header("Settings")
        selected_model_key = st.selectbox("Select Model", options=MODEL_KEYS)
        st.session_state.model_key = selected_model_key
        st.markdown("---")
        if st.button("➕ Start New Chat", use_container_width=True):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        st.caption(f"Session: {st.session_state.conversation_id[:8]}...")
        st.caption(f"User: {st.session_state.user_id[:8]}...")
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.auth_token = None
            st.session_state.user_id = None
            st.rerun()

    # --- API Call Function ---
    def call_api_and_get_response(prompt_text):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.auth_token}" # <-- SEND THE TOKEN
        }
        payload = {
            "conversation_id": st.session_state.conversation_id,
            "model_key": st.session_state.model_key,
            "message": prompt_text,
        }
        try:
            response = requests.post(f"{API_BASE_URL}/chat", headers=headers, data=json.dumps(payload))
            if response.status_code == 401:
                 st.error("Authentication failed. Your session may have expired. Please log out and log back in.")
                 return "Authentication Error", "ERROR"
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Error: No text in response."), data.get("mode", "UNKNOWN")
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return "Server connection failed.", "ERROR"

    # --- Chat Interface ---
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