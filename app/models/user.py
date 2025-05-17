"""
User database model for Luna AI
Consolidated from multiple user model definitions
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
# Add the missing import for relationship
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    """
    User model representing a user in the system
    Used for authentication and video ownership
    """
    __tablename__ = "users"
    
    # Primary key is a UUID stored as a string
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User identification fields
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    bio = Column(String, nullable=True)
    
    # Authentication fields
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    videos = relationship("Video", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        """String representation of the user"""
        return f"<User {self.email}>"