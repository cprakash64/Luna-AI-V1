"""
Chat database model for Luna AI
Stores chat conversations and interactions
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Chat(Base):
    """
    Chat model representing a Q&A interaction with the AI assistant
    Stores questions, answers, and timestamps
    """
    __tablename__ = "chats"
    
    # Primary key is a UUID stored as a string
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Chat content
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    tab_id = Column(String(255), nullable=True)  # For tracking session
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=True)
    
    # User and video relationships
    user = relationship("User")
    video = relationship("Video")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        """String representation of the chat"""
        return f"<Chat {self.id} - Q: {self.question[:20]}...>"


class ResetToken(Base):
    """
    ResetToken model for password reset functionality
    Tracks tokens, expiration, and usage
    """
    __tablename__ = "reset_tokens"
    
    # Primary key is a UUID stored as a string
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Token information
    token = Column(String(255), unique=True, index=True, nullable=False)
    is_used = Column(Boolean, default=False)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationship to user
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        """String representation of the reset token"""
        return f"<ResetToken {self.id} for User {self.user_id}>"