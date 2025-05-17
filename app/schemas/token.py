"""
Token schemas for Luna AI authentication
"""
from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    """
    Token schema for JWT authentication response
    
    Attributes:
        access_token: JWT token string
        token_type: Token type (bearer)
    """
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    """
    Token payload for JWT decoding
    
    Attributes:
        sub: Subject of the token (user ID)
        exp: Expiration timestamp
    """
    sub: Optional[str] = None
    exp: Optional[int] = None