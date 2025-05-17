# app/services/auth.py contains business logic and database operations
"""
Authentication services for Luna AI
Contains business logic for user authentication
"""
from datetime import timedelta, datetime
from typing import Any, Dict, Optional
import logging
import uuid
import os
from sqlalchemy.orm import Session
from sqlalchemy import text
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from app.utils.security import verify_password, get_password_hash, create_access_token
from app.config import settings
from app.utils.database import get_db

logger = logging.getLogger("auth_service")

# Create OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Check if we're in development mode
DEV_MODE = os.environ.get("DEV_MODE", "False").lower() in ("true", "1", "t")
logger.info(f"Auth service running in {'development' if DEV_MODE else 'production'} mode")

def authenticate_user(db: Session, email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by email and password
    Returns user data with tokens if authentication succeeds
    """
    try:
        logger.info(f"Authentication attempt for email: {email}")
        
        # Get user by email with direct SQL
        result = db.execute(
            text("SELECT id, email, hashed_password, is_active FROM users WHERE email = :email"),
            {"email": email}
        )
        user_row = result.fetchone()
        
        if not user_row:
            logger.warning(f"User with email {email} not found")
            return None
        
        user_id, user_email, hashed_password, is_active = user_row
        
        if not verify_password(password, hashed_password):
            logger.warning(f"Invalid password for user {email}")
            return None
            
        if not is_active:
            logger.warning(f"Inactive user {email} attempted login")
            return None
            
        # Create access token
        access_token = create_access_token(subject=str(user_id))
        
        # Create refresh token with longer expiry
        refresh_token = create_access_token(
            subject=str(user_id),
            expires_delta=timedelta(days=7)  # Longer expiry for refresh token
        )
        
        # Fetch additional user data
        user_result = db.execute(
            text("SELECT full_name, username FROM users WHERE id = :id"),
            {"id": user_id}
        )
        user_data = user_result.fetchone()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_id": user_id,
            "email": user_email,
            "fullname": user_data[0] if user_data else None,
            "username": user_data[1] if user_data else None
        }
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return None

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Get the current authenticated user from the JWT token
    This function is used as a dependency in protected routes
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Get user from database
        user = get_user_by_id(db, user_id)
        if user is None:
            raise credentials_exception
        
        # Check if user is active
        if not user.get("is_active", False):
            raise HTTPException(status_code=400, detail="Inactive user")
            
        return user
    except JWTError:
        logger.warning("Invalid authentication token")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Error in get_current_user: {str(e)}")
        raise credentials_exception

async def get_optional_auth(request: Request = None, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Optional authentication - use for endpoints that should work in dev mode without auth
    In development mode, it returns a default user if no valid token is provided
    In production mode, it works the same as get_current_user
    """
    # Check if we should bypass auth (for development only)
    if DEV_MODE:
        try:
            # Try to get the user if authenticated
            if token:
                try:
                    # Decode the JWT token
                    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
                    user_id = payload.get("sub")
                    if user_id:
                        user = get_user_by_id(db, user_id)
                        if user:
                            return user
                except (JWTError, Exception) as e:
                    logger.warning(f"Token validation failed in dev mode: {str(e)}")
            
            # If no token or invalid token in dev mode, provide a default testing user
            logger.warning("Auth bypassed in development mode - using default dev user")
            return {
                "id": "dev-user",
                "email": "dev@example.com",
                "username": "dev_user",
                "full_name": "Development User",
                "is_active": True,
                "authenticated": True
            }
        except Exception as e:
            logger.error(f"Error in get_optional_auth dev mode: {str(e)}")
            # Still return a dev user on error in dev mode
            return {
                "id": "dev-user",
                "email": "dev@example.com",
                "username": "dev_user",
                "full_name": "Development User",
                "is_active": True,
                "authenticated": True
            }
    else:
        # In production, use normal auth
        return await get_current_user(token, db)

def refresh_auth_token(db: Session, refresh_token: str) -> Optional[Dict[str, Any]]:
    """
    Refresh access token using a refresh token
    Returns a new access token if the refresh token is valid
    """
    try:
        # Decode and validate refresh token
        payload = jwt.decode(
            refresh_token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Extract user ID from token
        user_id = payload.get("sub")
        if not user_id:
            logger.warning("Refresh token missing sub claim")
            return None
        
        # Verify user exists
        result = db.execute(
            text("SELECT id, email, is_active FROM users WHERE id = :id"),
            {"id": user_id}
        )
        user_row = result.fetchone()
        
        if not user_row or not user_row[2]:  # Check if user exists and is active
            logger.warning(f"User {user_id} not found or inactive")
            return None
        
        # Create new access token
        access_token = create_access_token(subject=user_id)
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return None

def create_new_user(db: Session, email: str, fullname: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Create a new user
    Returns the created user's data if successful
    """
    try:
        logger.info(f"Creating new user with email: {email}")
        
        # Check if user exists
        result = db.execute(text("SELECT id FROM users WHERE email = :email"), {"email": email})
        existing_user = result.fetchone()
        if existing_user:
            logger.warning(f"User with email {email} already exists")
            return None
        
        # Check if username exists
        result = db.execute(text("SELECT id FROM users WHERE username = :username"), {"username": fullname})
        existing_username = result.fetchone()
        if existing_username:
            logger.warning(f"User with username {fullname} already exists")
            return None
        
        # Hash password
        hashed_password = get_password_hash(password)
        
        # Generate user ID
        user_id = str(uuid.uuid4())
        
        # Create timestamp
        timestamp = datetime.utcnow().isoformat()
        
        # Insert user into database
        db.execute(
            text("""
            INSERT INTO users 
            (id, email, username, hashed_password, full_name, is_active, created_at, updated_at) 
            VALUES (:id, :email, :username, :hashed_password, :full_name, :is_active, :created_at, :updated_at)
            """),
            {
                "id": user_id,
                "email": email,
                "username": fullname,
                "hashed_password": hashed_password,
                "full_name": fullname,
                "is_active": True,
                "created_at": timestamp,
                "updated_at": timestamp
            }
        )
        db.commit()
        
        return {
            "id": user_id,
            "email": email,
            "username": fullname
        }
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        if 'db' in locals():
            db.rollback()
        return None

def get_user_by_id(db: Session, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user by ID
    Returns user data if found
    """
    try:
        result = db.execute(
            text("SELECT id, email, username, full_name, is_active FROM users WHERE id = :id"),
            {"id": user_id}
        )
        user_row = result.fetchone()
        
        if not user_row:
            return None
        
        return {
            "id": user_row[0],
            "email": user_row[1],
            "username": user_row[2],
            "full_name": user_row[3],
            "is_active": user_row[4],
            "authenticated": True
        }
    except Exception as e:
        logger.error(f"Error getting user by ID: {str(e)}")
        return None

def verify_reset_token(token: str, salt: str = 'reset-password-salt', max_age: int = 3600) -> Optional[str]:
    """
    Verify a password reset token
    Returns the email associated with the token if valid
    """
    try:
        from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
        
        s = URLSafeTimedSerializer(settings.SECRET_KEY)
        email = s.loads(token, salt=salt, max_age=max_age)
        return email
    except Exception as e:
        logger.error(f"Error verifying reset token: {str(e)}")
        return None

def generate_reset_token(email: str, salt: str = 'reset-password-salt') -> str:
    """
    Generate a password reset token
    Returns the token
    """
    from itsdangerous import URLSafeTimedSerializer
    
    s = URLSafeTimedSerializer(settings.SECRET_KEY)
    return s.dumps(email, salt=salt)

def reset_user_password(db: Session, email: str, new_password: str) -> bool:
    """
    Reset a user's password
    Returns True if successful
    """
    try:
        # Find user
        result = db.execute(text("SELECT id FROM users WHERE email = :email"), {"email": email})
        user_row = result.fetchone()
        
        if not user_row:
            return False
        
        # Update password
        hashed_password = get_password_hash(new_password)
        db.execute(
            text("UPDATE users SET hashed_password = :password WHERE email = :email"),
            {"password": hashed_password, "email": email}
        )
        db.commit()
        
        return True
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        if 'db' in locals():
            db.rollback()
        return False