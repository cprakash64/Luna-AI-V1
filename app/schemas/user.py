"""
User schemas for Luna AI
Handles request/response validation for user operations
"""
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, validator

class UserBase(BaseModel):
    """
    Base user schema with common attributes
    
    Attributes:
        email: User's email address
        username: User's username
    """
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None

class UserCreate(BaseModel):
    """
    Schema for user creation requests
    
    Attributes:
        email: User's email address (required)
        username: User's username (required)
        password: User's password (required)
    """
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    # Validator to ensure password meets requirements
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserUpdate(UserBase):
    """
    Schema for user update requests
    
    Attributes:
        password: Optional new password
    """
    password: Optional[str] = None
    
    # Validator to ensure password meets requirements if provided
    @validator('password')
    def password_strength(cls, v):
        if v is not None and len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserInDBBase(UserBase):
    """
    Base schema for user from database
    
    Attributes:
        id: User's ID (UUID)
    """
    id: UUID
    
    class Config:
        from_attributes = True  # Updated from orm_mode = True for Pydantic v2

class User(UserInDBBase):
    """
    Schema for user response
    Inherits all fields from UserInDBBase
    """
    pass

class UserInDB(UserInDBBase):
    """
    Schema for user in database
    
    Attributes:
        hashed_password: Hashed password string
    """
    hashed_password: str