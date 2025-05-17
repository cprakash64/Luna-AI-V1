"""
Video schemas for Luna AI
Handles request/response validation for video operations
"""
from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class VideoBase(BaseModel):
    """
    Base video schema with common attributes
    
    Attributes:
        title: Video title
        description: Optional video description
    """
    title: Optional[str] = None
    description: Optional[str] = None

class VideoCreate(VideoBase):
    """
    Schema for video creation requests
    
    Attributes:
        user_id: ID of the user who owns the video
    """
    user_id: UUID
    title: str = Field(..., min_length=1, max_length=255)

class VideoUpdate(VideoBase):
    """
    Schema for video update requests
    """
    status: Optional[str] = None
    processed: Optional[bool] = None
    transcription: Optional[str] = None
    summary: Optional[str] = None

class VideoInDBBase(VideoBase):
    """
    Base schema for video from database
    
    Attributes:
        id: Video ID (UUID)
        user_id: ID of the user who owns the video
        created_at: Timestamp when the video was created
        updated_at: Timestamp when the video was last updated
    """
    id: UUID
    user_id: UUID
    video_name: str
    video_url: str
    processed: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True  # Updated from orm_mode = True for Pydantic v2

class Video(VideoInDBBase):
    """
    Schema for video response
    Inherits all fields from VideoInDBBase
    """
    pass

class VideoInDB(VideoInDBBase):
    """
    Schema for video in database
    Contains additional internal fields
    """
    status: Optional[str] = "pending"
    transcription: Optional[str] = None
    summary: Optional[str] = None