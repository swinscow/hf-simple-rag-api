# auth.py - FINAL CODE FOR SUPABASE AUTHENTICATION

from fastapi import Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from jose import jwt
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# CRITICAL: Get these from your Supabase Project Settings > API > URL and Key
SUPABASE_URL = os.getenv("SUPABASE_URL") 
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY not found. Check Render ENV!")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
token_auth_scheme = HTTPBearer()

class User(BaseModel):
    user_id: str
    is_authenticated: bool = True

# --- CRITICAL AUTHENTICATION FUNCTION ---
def get_current_user(token: HTTPAuthorizationCredentials = Security(token_auth_scheme)) -> User:
    """
    Validates the JWT token using the Supabase client.
    Returns a User object containing the authenticated user's unique ID.
    """
    """
    TEMPORARY: Bypasses ALL authentication checks for cloud persistence testing.
    The real user_id should come from the validated token.
    """
    # Use a fixed, secure user ID for testing the database persistence
    # return User(user_id="SUPABASE_RAG_TESTER")

    if token.credentials is None:
        raise HTTPException(status_code=403, detail='Not authenticated. Token is missing.')

    try:
        # 1. Supabase validation: Verifies the token's signature, issuer, and expiry
        # This is the line that talks to the secure Supabase service internally
        # We assume the user has exchanged their login credentials for a JWT token previously.
        auth_response = supabase.auth.get_user(token.credentials)

        if auth_response is None or auth_response.user is None:
             raise HTTPException(status_code=401, detail='Invalid or expired token provided.')
        
        # 2. Success: Extract the user ID (Supabase stores this in the 'id' field)
        # Note: Auth0 used 'sub', Supabase uses 'id'
        user_id = auth_response.user.id
        
        return User(user_id=str(user_id))

    except Exception as e:
        # Catch network errors, token decoding issues, etc.
        print(f"Auth Error Detail: {e}")
        raise HTTPException(status_code=401, detail='Token validation failed.')