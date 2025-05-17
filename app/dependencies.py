"""
FastAPI dependencies for Luna AI
Provides common dependencies for API endpoints
"""
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from pydantic import ValidationError
from datetime import datetime
from app.utils.database import get_db, get_async_db
from app.config import settings
from app.schemas.token import TokenPayload
from app.models.user import User
from app.services.visual_analysis import get_visual_analysis_service

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login"
)

def get_current_user(
    db: Session = Depends(get_db), 
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Validate token and return current user - synchronous version
    
    Args:
        db: Database session
        token: JWT token from Authorization header
        
    Returns:
        User model
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        # Decode and validate token
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        # Check if token has expired
        if datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    from sqlalchemy import select
    result = db.execute(select(User).where(User.id == token_data.sub))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

async def get_current_user_async(
    db = Depends(get_async_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Validate token and return current user - asynchronous version
    
    Args:
        db: Async database session
        token: JWT token from Authorization header
        
    Returns:
        User model
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        # Decode and validate token
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        # Check if token has expired
        if datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database using async session
    from sqlalchemy import select
    stmt = select(User).where(User.id == token_data.sub)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

async def get_optional_current_user(
    db = Depends(get_async_db),
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None - asynchronous version
    
    Args:
        db: Async database session
        token: Optional JWT token from Authorization header
        
    Returns:
        User model or None
    """
    if token is None:
        return None
        
    try:
        return await get_current_user_async(db, token)
    except HTTPException:
        return None

def get_visual_analysis_service_dependency():
    """
    Visual analysis service dependency for FastAPI routes
    """
    return get_visual_analysis_service()