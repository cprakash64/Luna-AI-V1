"""
Models initialization for Luna AI
Exports all database models for use throughout the application
"""
from app.models.user import User, Base as UserBase
from app.models.video import Video, Frame, Base as VideoBase
from app.models.chat import Chat, ResetToken, Base as ChatBase

# Export a consistent Base for migrations and database initialization
Base = UserBase

# Export all models for convenience
__all__ = ["User", "Video", "Frame", "Chat", "ResetToken", "Base"]