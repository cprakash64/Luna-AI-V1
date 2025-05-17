"""
Schemas initialization for Luna AI
Exports all Pydantic schemas for use throughout the application
"""
from app.schemas.token import Token, TokenPayload
from app.schemas.user import UserBase, UserCreate, UserUpdate, User, UserInDB
from app.schemas.video import VideoBase, VideoCreate, VideoUpdate, Video, VideoInDB
from app.schemas.common import (
    AIQuestion, AIResponse,
    ObjectDetectionRequest, ObjectDetectionResult,
    TranscriptionRequest, TranscriptionResponse,
    UserProfileUpdate, PasswordChange, PasswordReset
)

# Export all schemas for convenience
__all__ = [
    # Token schemas
    "Token", "TokenPayload",
    
    # User schemas
    "UserBase", "UserCreate", "UserUpdate", "User", "UserInDB",
    "UserProfileUpdate", "PasswordChange", "PasswordReset",
    
    # Video schemas
    "VideoBase", "VideoCreate", "VideoUpdate", "Video", "VideoInDB",
    
    # AI and processing schemas
    "AIQuestion", "AIResponse",
    "ObjectDetectionRequest", "ObjectDetectionResult",
    "TranscriptionRequest", "TranscriptionResponse"
]