"""
Database utilities for Luna AI
Handles database connection, session management, and base operations
"""
import logging
from typing import Generator, Any, List
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import uuid

from app.config import settings
from app.models import Base

# Configure logging
logger = logging.getLogger("database")

# Convert PostgresDsn to string for SQLAlchemy
db_url = str(settings.DATABASE_URL)

# Create SQLAlchemy engine for synchronous operations
engine = create_engine(
    db_url,
    pool_pre_ping=True,  # Verify connection is still alive
    echo=settings.DB_ECHO,  # Log SQL queries if debug mode
)

# Create sessionmaker for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create SQLAlchemy engine for asynchronous operations
# Convert postgresql:// to postgresql+asyncpg:// for async connections
async_database_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
async_engine = create_async_engine(
    async_database_url,
    echo=settings.DB_ECHO,
)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Enable UUID extension for PostgreSQL
@event.listens_for(Engine, "connect")
def enable_uuid_extension(dbapi_connection, connection_record):
    """
    Enable UUID extension when connecting to PostgreSQL
    """
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        cursor.close()
        dbapi_connection.commit()
    except Exception as e:
        logger.warning(f"Could not enable uuid-ossp extension: {str(e)}")

def create_tables() -> None:
    """
    Create all tables defined in models
    Only used for initial setup or testing
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

def drop_tables() -> None:
    """
    Drop all tables defined in models
    Only used for cleanup or testing
    """
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped")

def get_db() -> Generator[Session, None, None]:
    """
    Create a new database session for each request
    Used as a FastAPI dependency for synchronous operations
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> Generator[AsyncSession, None, None]:
    """
    Create a new async database session for each request
    Used as a FastAPI dependency for asynchronous operations
    
    Yields:
        SQLAlchemy async session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions
    Useful for scripts and background tasks
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

async def list_tables() -> List[str]:
    """
    Get a list of all tables in the database
    
    Returns:
        List of table names
    """
    inspector = inspect(engine)
    return inspector.get_table_names()

def init_db() -> None:
    """
    Initialize database with required tables and seed data
    """
    try:
        # Create tables if they don't exist
        create_tables()
        
        # Check if we need to create initial data
        with get_db_context() as db:
            # Example: Check if admin user exists and create if not
            from app.models.user import User
            from app.utils.security import get_password_hash
            
            admin_email = "admin@example.com"
            admin_exists = db.query(User).filter(User.email == admin_email).first()
            
            if not admin_exists:
                admin_user = User(
                    id=uuid.uuid4(),
                    email=admin_email,
                    username="admin",
                    full_name="Admin User",
                    hashed_password=get_password_hash("admin123"),  # Use environment variable in production
                    is_active=True,
                    is_superuser=True,
                )
                db.add(admin_user)
                db.commit()
                logger.info("Admin user created")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise