# auth.py - The Clean, Minimal Structure (Relies on Environment Variables)

from fastapi import Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from jose import jwt
import os
import requests
from dotenv import load_dotenv

# Load environment variables just to ensure they are available for os.getenv
# NOTE: This load_dotenv() is often redundant on Render but good for local dev
load_dotenv()

# --- Configuration (Reads from your .env file) ---
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID") 
# We need to rely on the library reading these variables implicitly or via fetch
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")

# Define the security scheme and User model
token_auth_scheme = HTTPBearer()

class User(BaseModel):
    user_id: str
    is_authenticated: bool = True

# Global cache for Auth0's signing keys
jwks = None

# --- CRITICAL AUTHENTICATION FUNCTION ---
def get_current_user(token: HTTPAuthorizationCredentials = Security(token_auth_scheme)) -> User:
    """
    Validates the JWT token received in the 'Authorization: Bearer <token>' header.
    It relies on jose/requests to handle JWKS fetching and validation.
    """
    global jwks
    
    # Check 1: Token is present
    if token.credentials is None:
        raise HTTPException(status_code=403, detail='Not authenticated. Token is missing.')

    # 2. Fetch Signing Keys if not cached
    if jwks is None:
        try:
            jwks_uri = f'https://{AUTH0_DOMAIN}/.well-known/jwks.json'
            # Use requests to fetch the public key set (this is the SSL failure point)
            response = requests.get(jwks_uri, auth=(AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET))
            response.raise_for_status()
            jwks = response.json()
        except requests.RequestException:
             # Raise an internal server error if Auth0 service is unreachable
            raise HTTPException(status_code=500, detail='Authentication server error: Could not fetch JWKS.')

    unverified_header = jwt.get_unverified_header(token.credentials)
    
    # 3. Find the correct public key to decode the token
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

    # 4. Decode and Validate the Token
    try:
        payload = jwt.decode(
            token.credentials,
            rsa_key,
            algorithms=['RS256'],
            audience=AUTH0_CLIENT_ID, 
            issuer=f'https://{AUTH0_DOMAIN}/'
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Token is expired.')
    except jwt.JWTClaimsError:
        raise HTTPException(status_code=401, detail='Invalid claims.')
    except Exception:
        raise HTTPException(status_code=401, detail='Invalid token.')

    # 5. Success: Extract the user ID 
    user_id = payload['sub']
    
    return User(user_id=user_id)