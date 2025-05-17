"""
Security utilities for Luna AI
Handles password hashing, verification, and token management
"""
from typing import Optional, Union, Any
from datetime import datetime, timedelta
import logging
from passlib.context import CryptContext
from jose import jwt, JWTError

from app.config import settings

# Configure logging
logger = logging.getLogger("security")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password to compare against
        
    Returns:
        True if the password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)

def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token
    
    Args:
        subject: Subject of the token (usually user ID)
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token as string
    """
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    # Create JWT payload
    to_encode = {
        "exp": expire, 
        "sub": str(subject),
        "iat": datetime.utcnow()  # Issued at time
    }
    
    # Encode the JWT token
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt

def validate_token(token: str) -> Optional[dict]:
    """
    Validate a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Token payload dictionary if valid, None otherwise
    """
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Check if token has expired
        if datetime.fromtimestamp(payload.get("exp")) < datetime.utcnow():
            logger.warning("Token has expired")
            return None
            
        return payload
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        return None