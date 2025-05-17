"""
Repository services for Luna AI
Generic repository pattern implementation for database operations
"""
from sqlalchemy.orm import Session
from typing import Generic, TypeVar, Type, List, Optional, Union, Dict, Any
from uuid import UUID
from pydantic import BaseModel

from app.models import Base

# Define TypeVars for generic typing
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Generic repository base class providing standard database operations
    
    Attributes:
        model: The SQLAlchemy model class to use
    """
    def __init__(self, model: Type[ModelType]):
        """
        Initialize repository with a specific model
        
        Args:
            model: The SQLAlchemy model class
        """
        self.model = model

    def get(self, db: Session, id: UUID) -> Optional[ModelType]:
        """
        Get an object by ID
        
        Args:
            db: Database session
            id: Object ID
            
        Returns:
            Object instance or None if not found
        """
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        Get multiple objects with pagination
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of objects
        """
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        Create a new object
        
        Args:
            db: Database session
            obj_in: Object creation schema
            
        Returns:
            Created object instance
        """
        obj_in_data = obj_in.model_dump(exclude_unset=True)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, *, db_obj: ModelType, obj_in: Union[UpdateSchemaType, Dict[str, Any]]) -> ModelType:
        """
        Update an object
        
        Args:
            db: Database session
            db_obj: Object instance to update
            obj_in: Update schema or dictionary with fields to update
            
        Returns:
            Updated object instance
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
            
        for field in update_data:
            setattr(db_obj, field, update_data[field])
            
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: UUID) -> ModelType:
        """
        Delete an object by ID
        
        Args:
            db: Database session
            id: Object ID
            
        Returns:
            Deleted object instance
        """
        obj = db.query(self.model).get(id)
        db.delete(obj)
        db.commit()
        return obj

class UserRepository(BaseRepository):
    """
    User repository with specialized methods for user operations
    """
    def get_by_email(self, db: Session, *, email: str) -> Optional[ModelType]:
        """
        Get a user by email
        
        Args:
            db: Database session
            email: User's email
            
        Returns:
            User instance or None if not found
        """
        return db.query(self.model).filter(self.model.email == email).first()
    
    def get_by_username(self, db: Session, *, username: str) -> Optional[ModelType]:
        """
        Get a user by username
        
        Args:
            db: Database session
            username: User's username
            
        Returns:
            User instance or None if not found
        """
        return db.query(self.model).filter(self.model.username == username).first()
    
    def authenticate(self, db: Session, *, email: str, password: str) -> Optional[ModelType]:
        """
        Authenticate a user by email and password
        
        Args:
            db: Database session
            email: User's email
            password: User's password
            
        Returns:
            User instance if authenticated, None otherwise
        """
        from app.utils.security import verify_password
        
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

class VideoRepository(BaseRepository):
    """
    Video repository with specialized methods for video operations
    """
    def get_by_user_id(self, db: Session, *, user_id: UUID, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        Get videos by user ID
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of video objects
        """
        return db.query(self.model).filter(self.model.user_id == user_id).offset(skip).limit(limit).all()
    
    def get_by_title(self, db: Session, *, title: str) -> Optional[ModelType]:
        """
        Get a video by title
        
        Args:
            db: Database session
            title: Video title
            
        Returns:
            Video instance or None if not found
        """
        return db.query(self.model).filter(self.model.title == title).first()