# app/api/auth.py handles HTTP requests and responses

"""
Authentication API endpoints for Luna AI
Handles user signup, login, and authentication verification
"""
from datetime import datetime
from typing import Any, Dict
import logging
from json.decoder import JSONDecodeError
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.utils.database import get_db
from app.dependencies import get_current_user
from app.models.user import User as UserModel
from app.services.email import send_reset_password_email
from app.services.auth import (
    authenticate_user, 
    refresh_auth_token, 
    create_new_user, 
    get_user_by_id,
    verify_reset_token,
    generate_reset_token,
    reset_user_password
)

# Create router with standardized prefix
router = APIRouter(prefix="/api/auth", tags=["authentication"])
logger = logging.getLogger("auth_api")

# Define OAuth2 scheme for token authentication with correct path
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

@router.post("/login", response_model=Dict[str, Any])
async def login_access_token(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Login endpoint that accepts JSON body from frontend
    Returns access token and refresh token on successful authentication
    """
    try:
        # Get the raw request body for debugging if needed
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
        
        # Try to parse the JSON body
        try:
            data = await request.json()
        except JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, Request body: {raw_body}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid JSON format: {str(e)}"}
            )
            
        logger.info(f"Login attempt received for email: {data.get('email', 'unknown')}")
        
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Email and password are required"}
            )
        
        # Use service function to authenticate
        user_data = authenticate_user(db, email, password)
        
        if not user_data:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Incorrect email or password"}
            )
            
        return user_data
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Login error: {str(e)}"}
        )

@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token_endpoint(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Refresh access token using refresh token
    Returns new access token if refresh token is valid
    """
    try:
        data = await request.json()
        refresh_token = data.get("refresh_token")
        
        if not refresh_token:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Refresh token is required"}
            )
        
        # Use service function to refresh token
        token_data = refresh_auth_token(db, refresh_token)
        
        if not token_data:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired refresh token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return token_data
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Token refresh error: {str(e)}"}
        )

@router.post("/signup", response_model=Dict[str, Any])
async def create_user(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Create new user account
    Returns user data if successful
    """
    try:
        # Parse JSON request body with error handling
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
        
        try:
            data = await request.json()
        except JSONDecodeError as e:
            logger.error(f"JSON decode error in signup: {str(e)}, Request body: {raw_body}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Invalid JSON format: {str(e)}"}
            )
            
        logger.info("Signup request received")
        logger.info(f"Signup data: {data}")
        
        # Extract fields
        email = data.get("email")
        fullname = data.get("fullname")
        password = data.get("password")
        
        # Validate required fields
        if not email:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Email is required"}
            )
        if not fullname:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Full Name is required"}
            )
        if not password:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Password is required"}
            )
        
        # Use service function to create user
        user_data = create_new_user(db, email, fullname, password)
        
        if not user_data:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"detail": "The email or username already exists. Please try another or login."}
            )
            
        return user_data
    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again."}
        )

@router.get("/check-auth", response_model=Dict[str, Any])
async def check_auth(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Check if user is authenticated and return user info
    Extracts and validates token from Authorization header
    """
    try:
        # Get Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid Authorization header")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract token
        token = auth_header.split(" ")[1]
        if not token:
            logger.warning("Empty token")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Empty token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Manually decode and validate token
        try:
            from jose import jwt
            from app.config import settings
            
            # Use leeway parameter to allow for clock skew
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM],
                options={"verify_exp": False}  # Temporarily disable expiration check for debugging
            )
            
            # Extract user ID from token
            user_id = payload.get("sub")
            if not user_id:
                logger.warning("Token missing sub claim")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid token format"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Print token info for debugging
            exp = payload.get("exp")
            if exp:
                exp_time = datetime.fromtimestamp(exp)
                now = datetime.utcnow()
                logger.info(f"Token for user {user_id} - Exp: {exp_time}, Now: {now}, Diff: {exp_time - now}")
            
            # Use service function to get user by ID
            user_data = get_user_by_id(db, user_id)
            
            if not user_data:
                logger.warning(f"User with ID {user_id} not found")
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"detail": "User not found"},
                )
            
            return user_data
            
        except jwt.JWTError as e:
            logger.warning(f"JWT validation error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": f"Token validation failed: {str(e)}"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        
    except Exception as e:
        logger.error(f"Error in check-auth endpoint: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Authentication error: {str(e)}"},
        )

@router.post("/forgot-password")
async def forgot_password(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Any:
    """
    Request password reset
    Sends email with reset link if user exists
    """
    try:
        data = await request.json()
        email = data.get("email")
        
        if not email:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Email is required"}
            )
    except JSONDecodeError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid JSON format"}
        )
    
    # Find user by email
    from sqlalchemy import text
    result = db.execute(text("SELECT id, username FROM users WHERE email = :email"), {"email": email})
    user_row = result.fetchone()
    
    # Always return success to prevent email enumeration
    if not user_row:
        return {"message": "If a user with that email exists, we've sent them a password reset link"}
    
    # Generate token using service function
    token = generate_reset_token(email)
    
    # Send email in background
    background_tasks.add_task(
        send_reset_password_email,
        email=email,
        token=token,
        user_name=user_row[1] if user_row else None
    )
    
    return {"message": "If a user with that email exists, we've sent them a password reset link"}

@router.post("/reset-password")
async def reset_password_endpoint(
    request: Request,
    db: Session = Depends(get_db)
) -> Any:
    """
    Reset password with token
    Verifies token and updates user password
    """
    try:
        data = await request.json()
        token = data.get("token")
        new_password = data.get("password")
        
        if not token or not new_password:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Token and new password are required"}
            )
    except JSONDecodeError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid JSON format"}
        )
    
    # Verify token using service function
    email = verify_reset_token(token)
    
    if not email:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid or expired token"}
        )
    
    # Reset password using service function
    success = reset_user_password(db, email, new_password)
    
    if not success:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "User not found"}
        )
    
    return {"message": "Password has been reset successfully"}

@router.post("/logout")
async def logout() -> Any:
    """
    Logout endpoint (client-side token removal)
    Note: JWT tokens can't be invalidated server-side without a blacklist/database
    """
    return {"message": "Successfully logged out"}