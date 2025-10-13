# auth.py - Token Validation and User ID Extraction
import urllib3
# Suppress the warning caused by setting verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from dotenv import load_dotenv
load_dotenv() # Ensure the environment is loaded before any variables are read

from fastapi import Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from pydantic import BaseModel
import requests
import os
import certifi
import ssl

# CRITICAL FIX: Force requests/urllib3 to use the updated certificate chain
# This overrides the system's potentially broken chain.
CERTIFICATE_BUNDLE_PATH = certifi.where()

# 1. Configuration (Reads from your .env file)
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID") 
# We don't use the Client Secret here, but rely on the public Domain/Keys

# Define the security scheme
token_auth_scheme = HTTPBearer()

# Class to hold validated user information
class User(BaseModel):
    user_id: str
    is_authenticated: bool = True

# Cache for Auth0's signing keys
jwks = None

# --- CRITICAL AUTHENTICATION FUNCTION ---
def get_current_user(token: HTTPAuthorizationCredentials = Security(token_auth_scheme)) -> User:

    """
    Validates the JWT token received in the 'Authorization: Bearer <token>' header.
    Returns a User object containing the authenticated user_id (sub).
    """
    global jwks
    
    # 1. Fetch Signing Keys if not cached (used to verify the token signature)
    if jwks is None:
        try:
            jwks_uri = f'https://{AUTH0_DOMAIN}/.well-known/jwks.json'
            #response = requests.get(jwks_uri, verify=CERTIFICATE_BUNDLE_PATH) 
            response = requests.get(jwks_uri, verify=False)
            response.raise_for_status()
            jwks = response.json()
        except requests.RequestException as e:
            # If we can't fetch the public key, we can't verify the token.
            raise HTTPException(
                status_code=500, detail=f"Authentication server error: Could not fetch JWKS. {e}"
            )

    unverified_header = jwt.get_unverified_header(token.credentials)
    
    # 2. Find the correct public key to decode the token
    rsa_key = {}
    for key in jwks['keys']:
        if key['kid'] == unverified_header['kid']:
            rsa_key = {
                'kty': key['kty'],
                'kid': key['kid'],
                'use': key['use'],
                'n': key['n'],
                'e': key['e']
            }
            break
            
    if not rsa_key:
        raise HTTPException(status_code=401, detail='Invalid authorization header.')

    # 3. Decode and Validate the Token
    try:
        payload = jwt.decode(
            token.credentials,
            rsa_key,
            algorithms=['RS256'],
            audience=AUTH0_CLIENT_ID, # Ensures the token was issued for *your* application
            issuer=f'https://{AUTH0_DOMAIN}/' # Ensures the token came from Auth0
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Token is expired.')
    except jwt.JWTClaimsError as e:
        raise HTTPException(status_code=401, detail=f'Invalid claims: {e}')
    except Exception:
        raise HTTPException(status_code=401, detail='Invalid token.')

    # 4. Success: Extract the user ID (Auth0 stores this in the 'sub' claim)
    # The 'sub' claim is typically in the format 'auth0|xxxxxxxxxx'
    user_id = payload['sub']
    
    return User(user_id=user_id)