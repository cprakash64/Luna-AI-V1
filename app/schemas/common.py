"""
Common schemas for Luna AI
Contains shared schemas used across different API endpoints
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

# User profile schemas
class UserProfileUpdate(BaseModel):
    """
    Schema for user profile updates
    
    Attributes:
        fullname: User's full name
    """
    fullname: Optional[str] = None
    
    @validator('fullname')
    def validate_fullname(cls, v):
        if v is not None and (len(v) < 3 or len(v) > 100):
            raise ValueError('Fullname must be between 3 and 100 characters')
        return v

class PasswordChange(BaseModel):
    """
    Schema for password change requests
    
    Attributes:
        current_password: User's current password
        new_password: New password (min length 8)
        confirm_password: Confirmation of new password
    """
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v

class PasswordReset(BaseModel):
    """
    Schema for password reset requests
    
    Attributes:
        token: Password reset token
        new_password: New password (min length 8)
        confirm_password: Confirmation of new password
    """
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

# Video upload and processing schemas
class VideoUpload(BaseModel):
    """
    Schema for video upload requests
    
    Attributes:
        tab_id: Client tab ID for tracking
    """
    tab_id: str

class VideoProcess(BaseModel):
    """
    Schema for video processing requests
    
    Attributes:
        youtube_url: YouTube URL to process
        tab_id: Client tab ID for tracking
    """
    youtube_url: str
    tab_id: str

# AI and processing schemas
class AIQuestion(BaseModel):
    """
    Schema for AI question requests
    
    Attributes:
        question: User's question
        video_id: ID of the video to analyze
        tab_id: Optional client tab ID
    """
    question: str
    video_id: UUID
    tab_id: Optional[str] = None

class AIResponse(BaseModel):
    """
    Schema for AI response
    
    Attributes:
        answer: AI's answer to the question
        transcription_segment: Optional relevant transcript segment
        frame_text: Optional text from relevant frame
        frame_description: Optional description of relevant frame
        frame_path: Optional path to relevant frame image
        timestamp: Optional timestamp for the response
    """
    answer: str
    transcription_segment: Optional[str] = None
    frame_text: Optional[str] = None
    frame_description: Optional[str] = None
    frame_path: Optional[str] = None
    timestamp: Optional[float] = None

class ObjectDetectionRequest(BaseModel):
    """
    Schema for object detection requests
    
    Attributes:
        video_id: ID of the video to analyze
        target_object: Object to detect
    """
    video_id: UUID
    target_object: str

class ObjectDetectionResult(BaseModel):
    """
    Schema for object detection results
    
    Attributes:
        timestamp: Timestamp where object was detected
        label: Object label
    """
    timestamp: float
    label: str

class TranscriptionRequest(BaseModel):
    """
    Schema for transcription requests
    
    Attributes:
        video_id: ID of the video to transcribe
        tab_id: Optional client tab ID
    """
    video_id: UUID
    tab_id: Optional[str] = None

class TranscriptionResponse(BaseModel):
    """
    Schema for transcription response
    
    Attributes:
        transcription: Transcribed text
    """
    transcription: str