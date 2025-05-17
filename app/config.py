# app/config.py
"""
Configuration settings for Luna AI application
Updated for Pydantic 2.x with enhanced environment variable handling
"""
import os
import secrets
import logging
from pathlib import Path
from typing import Optional
from pydantic import field_validator, PostgresDsn
from pydantic_settings import BaseSettings

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config")

# Base directory for the application
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application settings with environment variable loading and validation"""
    
    # Project metadata
    PROJECT_NAME: str = "Luna AI"
    
    # API configuration
    API_V1_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"  # Added algorithm for JWT
    
    # CORS Settings - Parse as a string, we'll split it in the application
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
    
    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_DB: str
    DATABASE_URL: Optional[PostgresDsn] = None
    DB_ECHO: bool = False  # SQL query logging
    
    # Storage paths
    STORAGE_PATH: str = "./data/uploads"
    UPLOAD_DIR: Optional[Path] = None  # Will be set after initialization
    FRAME_DIR: Optional[Path] = None   # Will be set after initialization
    TRANSCRIPTION_DIR: Optional[Path] = None  # Will be set after initialization
    TEMP_DIR: Optional[Path] = None  # Will be set after initialization
    LOG_DIR: Optional[Path] = None  # Will be set after initialization
    
    # Frontend URL for links in emails, etc.
    FRONTEND_URL: str = "http://localhost:5173"
    
    # SMTP Configuration
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_TLS: bool = True
    SMTP_ENABLED: bool = False
    EMAILS_FROM_EMAIL: str = "noreply@luna-ai.example.com"
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: Optional[str] = None
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "localhost"
    MAIL_FROM_NAME: str = "Luna AI"
    MAIL_USE_TLS: bool = True
    MAIL_USE_SSL: bool = False
    
    # Environment
    ENVIRONMENT: str = "development"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Speech-to-Text API credentials
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    ASSEMBLYAI_API_KEY: Optional[str] = None
    
    # Gemini API Key for visual analysis
    GEMINI_API_KEY: Optional[str] = None
    
    # Socket.IO settings
    SOCKET_PING_TIMEOUT: int = 60
    SOCKET_PING_INTERVAL: int = 25
    
    # Feature flags
    AUTO_VISUAL_ANALYSIS: bool = False  # Add this field for visual analysis feature
    
    # Allow extra fields
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_url(cls, v: Optional[str], info) -> str:
        """Build DATABASE_URL from components if not provided"""
        if v:
            return v
        
        values = info.data
        
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB')}",
        )

# Create global settings instance
settings = Settings()

# Create storage directories with improved structure
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
TRANSCRIPTION_DIR = BASE_DIR / "data" / "transcriptions"
FRAME_DIR = BASE_DIR / "data" / "frames"
TEMP_DIR = BASE_DIR / "data" / "temp"
LOG_DIR = BASE_DIR / "data" / "logs"

# Create all directories
for directory in [UPLOAD_DIR, TRANSCRIPTION_DIR, FRAME_DIR, TEMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# Create important subdirectories
TIMESTAMPS_DIR = TRANSCRIPTION_DIR / "timestamps"
os.makedirs(TIMESTAMPS_DIR, exist_ok=True)
logger.info(f"Ensured timestamps directory exists: {TIMESTAMPS_DIR}")

# Add these paths to settings
settings.UPLOAD_DIR = UPLOAD_DIR
settings.TRANSCRIPTION_DIR = TRANSCRIPTION_DIR
settings.FRAME_DIR = FRAME_DIR
settings.TEMP_DIR = TEMP_DIR
settings.LOG_DIR = LOG_DIR

# Check and fix directory permissions
def check_directory_permissions():
    """Verify and fix permissions on key directories"""
    dirs_to_check = [
        UPLOAD_DIR,
        TRANSCRIPTION_DIR,
        TIMESTAMPS_DIR,
        FRAME_DIR,
        TEMP_DIR,
        LOG_DIR
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")
                continue
        
        # Check write permissions
        try:
            test_file = os.path.join(directory, ".permission_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Directory {directory} is writable")
        except Exception as e:
            logger.error(f"Directory {directory} is not writable: {str(e)}")
            try:
                # Try to fix permissions
                import stat
                os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)  # 0775
                logger.info(f"Attempted to fix permissions on {directory}")
            except Exception as fix_error:
                logger.error(f"Could not fix permissions: {str(fix_error)}")

# Run the permission check
check_directory_permissions()

# Set environment variables for API credentials
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
    logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS environment variable to {settings.GOOGLE_APPLICATION_CREDENTIALS}")
if settings.ASSEMBLYAI_API_KEY:
    os.environ["ASSEMBLYAI_API_KEY"] = settings.ASSEMBLYAI_API_KEY
    logger.info(f"Set ASSEMBLYAI_API_KEY environment variable (length: {len(settings.ASSEMBLYAI_API_KEY)})")
if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    logger.info(f"Set GEMINI_API_KEY environment variable (length: {len(settings.GEMINI_API_KEY)})")

# Set DEBUG_TRANSCRIPTION environment variable - this controls whether to use actual transcription
# services or debug (mock) mode. Set to "1" to use debug mode.
os.environ["DEBUG_TRANSCRIPTION"] = os.environ.get("DEBUG_TRANSCRIPTION", "0")
logger.info(f"DEBUG_TRANSCRIPTION is set to: {os.environ['DEBUG_TRANSCRIPTION']}")

# os.environ["DEBUG_TRANSCRIPTION"] = "1"
# logger.info("DEBUG_TRANSCRIPTION has been enabled to ensure uploads work")