"""
Video and Frame database models for Luna AI
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, Float, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Video(Base):
    """
    Video model representing a processed or uploaded video
    Stores video metadata and processing state
    """
    __tablename__ = "videos"
    
    # Primary key is a UUID stored as a string
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Video information
    video_name = Column(String(255), nullable=False)
    video_url = Column(String, nullable=False)  # Either file path or YouTube URL
    title = Column(String, index=True)
    description = Column(Text, nullable=True)
    
    # Processing information
    processed = Column(Boolean, default=False)
    status = Column(String, default="pending")  # pending, processing, completed, error
    duration = Column(Float, nullable=True)  # Video duration in seconds
    transcription = Column(Text, nullable=True)  # Full video transcription
    summary = Column(Text, nullable=True)
    analysis = Column(JSONB, nullable=True)  # JSON blob with analysis data
    tab_id = Column(String(255), nullable=True)
    
    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="videos")
    frames = relationship("Frame", back_populates="video", cascade="all, delete-orphan")
    
    def __repr__(self):
        """String representation of the video"""
        return f"<Video {self.video_name}>"


class Frame(Base):
    """
    Frame model representing extracted key frames from videos
    Stores frame data and analysis results
    """
    __tablename__ = "frames"
    
    # Primary key is a UUID stored as a string
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Frame information
    frame_path = Column(String, nullable=False)  # Path to the frame image file
    timestamp = Column(Float, nullable=False)  # Timestamp in seconds
    
    # Analysis results
    objects = Column(JSONB, nullable=True)  # Detected objects in the frame
    detected_objects = Column(Text, nullable=True)  # String version of detected objects
    text = Column(Text, nullable=True)  # Extracted text via OCR
    ocr_text = Column(Text, nullable=True)  # Alternative OCR text field
    faces = Column(JSONB, nullable=True)  # Detected faces and expressions
    
    # Relationship to video
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    video = relationship("Video", back_populates="frames")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        """String representation of the frame"""
        return f"<Frame {self.timestamp}s from Video {self.video_id}>"