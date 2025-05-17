"""
Main application entry point for Luna AI
Initializes FastAPI app and includes all routes
"""
import logging
import os
import re
import time
import json
import uuid
import shutil
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, APIRouter, HTTPException, Depends, Query, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables properly
from dotenv import load_dotenv
load_dotenv()  # This ensures .env is loaded before any value is accessed

# FIX: Default to production mode (0) instead of debug mode (1)
# In production, we would want to default to 0 if not specified
DEBUG_MODE = os.environ.get("DEBUG_TRANSCRIPTION", "0")
os.environ["DEBUG_TRANSCRIPTION"] = DEBUG_MODE

# Import application modules
from app.api import api_router
from app.config import settings
from app.utils.database import create_tables, get_db
from app.utils.storage import ensure_directories_exist

# Import transcription service shutdown function
from app.services.transcription_service import (
    shutdown as transcription_service_shutdown,
    process_youtube_video,
    process_audio_file
)

# Import visual analysis modules
from app.api import object_detection, videos_visual
from app.services.visual_analysis import get_visual_analysis_service
from app.services.object_detection import get_detector
from app.services.clip_service import get_clip_service

# Import auth helpers
from app.services.auth import get_optional_auth

# Configure logging with more detailed format for production
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(name)s - %(process)d - %(pathname)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("app")

# Define a middleware for request timing
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application
    Handles startup and shutdown events
    """
    # Startup actions
    logger.info("Luna AI starting up...")
    
    # Log important environment variables
    logger.info(f"Environment variables loaded:")
    logger.info(f"  DEBUG_TRANSCRIPTION = {os.environ.get('DEBUG_TRANSCRIPTION')}")
    logger.info(f"  GOOGLE_APPLICATION_CREDENTIALS = {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    
    assemblyai_key = os.environ.get('ASSEMBLYAI_API_KEY', '')
    logger.info(f"  ASSEMBLYAI_API_KEY available: {bool(assemblyai_key)}")
    
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    logger.info(f"  GEMINI_API_KEY available: {bool(gemini_key)}")
    
    # Check transcription service availability
    from app.services.transcription import check_google_credentials, check_assemblyai_credentials
    
    google_available = check_google_credentials()
    assemblyai_available = check_assemblyai_credentials()
    
    logger.info(f"Transcription service availability:")
    logger.info(f"  Google Cloud Speech: {'AVAILABLE' if google_available else 'NOT AVAILABLE'}")
    logger.info(f"  AssemblyAI: {'AVAILABLE' if assemblyai_available else 'NOT AVAILABLE'}")
    
    if not google_available and not assemblyai_available:
        logger.warning("⚠️ NO TRANSCRIPTION SERVICES AVAILABLE - SYSTEM WILL USE DEBUG MODE ⚠️")
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory created at {upload_dir}")
    
    # Initialize Visual Analysis Services
    try:
        # Initialize object detector
        detector = get_detector()
        logger.info(f"Object detector initialized: {detector is not None}")
        
        # Initialize CLIP service
        clip_service = get_clip_service()
        logger.info(f"CLIP service available: {clip_service.is_available()}")
        
        # Initialize visual analysis service
        visual_service = get_visual_analysis_service()
        logger.info(f"Visual analysis service initialized: {visual_service is not None}")
    except Exception as e:
        logger.error(f"Error initializing visual analysis services: {str(e)}")
        logger.warning("⚠️ VISUAL ANALYSIS SERVICES MAY BE LIMITED ⚠️")
    
    # Create database tables
    create_tables()
    
    # Ensure storage directories exist
    ensure_directories_exist()
    
    # Create additional directories for visual analysis
    os.makedirs(settings.FRAME_DIR, exist_ok=True)
    
    # Ensure AUTO_VISUAL_ANALYSIS is enabled by default
    if not hasattr(settings, 'AUTO_VISUAL_ANALYSIS'):
        setattr(settings, 'AUTO_VISUAL_ANALYSIS', True)
        logger.info("AUTO_VISUAL_ANALYSIS set to True by default")
    
    # Set DEV_MODE environment variable for authentication bypass if needed
    # In production, we'd want this to be False
    if os.environ.get("PRODUCTION", "False").lower() == "true":
        os.environ["DEV_MODE"] = "False"
        logger.info("Production mode enabled - authentication required")
    else:
        os.environ["DEV_MODE"] = "True"
        logger.info("DEV_MODE enabled for authentication bypass")
    
    yield
    
    # Shutdown actions
    logger.info("Luna AI shutting down...")
    
    # Clean up transcription service resources
    try:
        logger.info("Shutting down transcription service...")
        await transcription_service_shutdown()
    except Exception as e:
        logger.error(f"Error shutting down transcription service: {str(e)}")

# Define allowed origins explicitly - in production, this should be restricted
# to your specific domains
ALLOWED_ORIGINS = [
    "http://localhost:5173", 
    "http://127.0.0.1:5173", 
    "http://localhost:8000", 
    "http://127.0.0.1:8000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    # Add your production domains here
    # "https://yourdomain.com",
]

# In production, use environment variables to configure origins
PROD_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "")
if PROD_ORIGINS:
    # Split by comma and add to allowed origins
    ALLOWED_ORIGINS.extend([origin.strip() for origin in PROD_ORIGINS.split(",")])

logger.info(f"Using CORS allowed origins: {ALLOWED_ORIGINS}")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Luna AI - Video Analysis Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# IMPORTANT: Import Socket.IO components first before any CORS middleware
# to prevent potential conflicts
from app.api.websockets import socket_app, sio, setup_socketio

# Update Socket.IO CORS settings
# This needs to happen before mounting
sio.cors_allowed_origins = ALLOWED_ORIGINS
logger.info(f"Updated Socket.IO CORS settings with allowed origins: {ALLOWED_ORIGINS}")

# Configure middlewares for FastAPI - Order matters!
# Add timing middleware
app.add_middleware(TimingMiddleware)

# Add GZip compression for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],  # Expose timing header
    max_age=3600  # Cache preflight requests for 1 hour
)
logger.info("CORS middleware configured with specific frontend origins")

# Setup Socket.IO with FastAPI app
logger.info("Setting up Socket.IO")
setup_socketio(app)

# Configure OAuth2 for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)

# Include API router with v1 prefix
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Include Visual Analysis API routers
app.include_router(object_detection.router, prefix="/api/v1", tags=["Object Detection"])
app.include_router(videos_visual.router, prefix="/api/v1", tags=["Visual Analysis"])

# Also include routes at paths expected by the frontend
from app.api.auth import router as auth_router
frontend_auth_router = APIRouter()
frontend_auth_router.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(frontend_auth_router, prefix="/api")

# Helper function to read timestamps from file
async def read_timestamps_file(video_id: str) -> dict:
    """Read timestamps from JSON file"""
    try:
        # Check if timestamps file exists
        timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
        if os.path.exists(timestamp_path):
            with open(timestamp_path, "r") as f:
                return json.load(f)
        return {"formatted_timestamps": [], "raw_timestamps": []}
    except Exception as e:
        logger.error(f"Error reading timestamps file: {str(e)}")
        return {"formatted_timestamps": [], "raw_timestamps": []}

# API endpoint to process YouTube URL
@app.get("/process-url")
async def process_url(youtubeUrl: str, tabId: str, token: str = Depends(oauth2_scheme)):
    """
    Process a YouTube URL
    """
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    logger.info(f"Processing YouTube URL: {youtubeUrl} for tabId: {tabId}")
    # Generate a unique video ID
    video_id = f"youtube_{int(time.time())}_{os.urandom(3).hex()}"
    logger.info(f"Generated new video ID: {video_id}")
    
    # Emit event to the Socket.IO client to handle via WebSockets
    sid = tabId  # We'll use tabId as the room/sid
    try:
        await sio.emit('transcription_status', {
            'status': 'received',
            'message': 'Request received and processing started...',
            'video_id': video_id
        }, room=sid)
        
        # Start processing the YouTube URL
        # Set output file for transcription
        transcription_dir = Path(settings.TRANSCRIPTION_DIR)
        transcription_dir.mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(transcription_dir, f"transcription_{video_id}.json")
        
        # Process the YouTube URL
        task_id = await process_youtube_video(
            youtube_url=youtubeUrl,
            session_id=sid,
            output_file=output_file,
            video_id=video_id
        )
        
        logger.info(f"YouTube processing task started with ID: {task_id}")
        
    except Exception as e:
        logger.error(f"Error processing YouTube URL: {e}", exc_info=True)
        await sio.emit('error', {
            'message': f"Error processing YouTube URL: {str(e)}",
            'video_id': video_id
        }, room=sid)
        return {"status": "error", "message": str(e)}
    
    return {
        "status": "success", 
        "message": "Processing started",
        "videoId": video_id
    }

# API endpoint to handle video uploads - improved version
@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    videoFile: UploadFile = File(...), 
    tabId: str = Form(...), 
    videoId: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    """
    Handle video file upload and trigger transcription
    """
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    if not videoFile:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    
    logger.info(f"Video file uploaded for tabId: {tabId}, filename: {videoFile.filename}")
    
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique video ID if not provided
        if not videoId:
            video_id = f"file_{int(time.time())}_{os.urandom(3).hex()}"
            logger.info(f"Generated new video ID for file upload: {video_id}")
        else:
            video_id = videoId
            logger.info(f"Using provided video ID: {video_id}")
        
        # Validate file extension for security
        file_extension = os.path.splitext(videoFile.filename)[1].lower()
        if file_extension not in ['.mp4', '.mov', '.avi', '.webm', '.mkv']:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {file_extension}"}
            )
            
        # Save the file with a unique name using the video_id to avoid conflicts
        unique_filename = f"{video_id}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save the uploaded file using a file object
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(videoFile.file, buffer)
        
        logger.info(f"Saved uploaded file to {file_path}")
        
        # Set output file for transcription
        transcription_dir = Path(settings.TRANSCRIPTION_DIR)
        transcription_dir.mkdir(parents=True, exist_ok=True)
        transcription_file = os.path.join(transcription_dir, f"transcription_{video_id}.json")
        
        # Notify the client we're starting the transcription
        try:
            await sio.emit('transcription_status', {
                'status': 'received',
                'message': 'File received, starting transcription...',
                'video_id': video_id
            }, room=tabId)
        except Exception as e:
            logger.error(f"Error emitting start notification: {e}")
        
        # Use the enhanced process_audio_file function (which can handle both video and audio)
        background_tasks.add_task(
            process_audio_file,
            file_path=file_path,
            session_id=tabId,
            output_file=transcription_file,
            file_id=video_id
        )
        
        return {
            "status": "success", 
            "message": "Upload complete, transcription started", 
            "videoId": video_id
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={"error": f"Failed to process upload: {str(e)}"}
        )

# API endpoint to handle audio file uploads
@app.post("/upload-audio")
async def upload_audio(
    background_tasks: BackgroundTasks,
    audioFile: UploadFile = File(...), 
    tabId: str = Form(...), 
    audioId: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    """
    Handle audio file upload and trigger transcription
    """
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    if not audioFile:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    
    logger.info(f"Audio file uploaded for tabId: {tabId}, filename: {audioFile.filename}")
    
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique audio ID if not provided
        if not audioId:
            audio_id = f"audio_{int(time.time())}_{os.urandom(3).hex()}"
            logger.info(f"Generated new audio ID for file upload: {audio_id}")
        else:
            audio_id = audioId
            logger.info(f"Using provided audio ID: {audio_id}")
        
        # Validate file extension for security
        file_extension = os.path.splitext(audioFile.filename)[1].lower()
        if file_extension not in ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported audio file type: {file_extension}"}
            )
            
        # Save the file with a unique name using the audio_id to avoid conflicts
        unique_filename = f"{audio_id}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save the uploaded file using a file object
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audioFile.file, buffer)
        
        logger.info(f"Saved uploaded audio file to {file_path}")
        
        # Set output file for transcription
        transcription_dir = Path(settings.TRANSCRIPTION_DIR)
        transcription_dir.mkdir(parents=True, exist_ok=True)
        transcription_file = os.path.join(transcription_dir, f"transcription_{audio_id}.json")
        
        # Notify the client we're starting the transcription
        try:
            await sio.emit('transcription_status', {
                'status': 'received',
                'message': 'Audio file received, starting transcription...',
                'video_id': audio_id  # Use video_id field for compatibility with frontend
            }, room=tabId)
        except Exception as e:
            logger.error(f"Error emitting start notification: {e}")
        
        # Use the enhanced process_audio_file function
        background_tasks.add_task(
            process_audio_file,
            file_path=file_path,
            session_id=tabId,
            output_file=transcription_file,
            file_id=audio_id
        )
        
        return {
            "status": "success", 
            "message": "Audio upload complete, transcription started", 
            "audioId": audio_id,
            "videoId": audio_id  # Include videoId for compatibility with frontend
        }
    except Exception as e:
        logger.error(f"Error processing uploaded audio file: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={"error": f"Failed to process audio upload: {str(e)}"}
        )

# API endpoint to get transcription
@app.get("/get-transcription")
async def get_transcription(
    tabId: str, 
    videoId: Optional[str] = None, 
    token: str = Depends(oauth2_scheme)
):
    """
    Get transcription for a video with improved ID handling
    """
    # Validate token in production mode - for this endpoint we'll make it optional
    # since it's mostly for retrieving data
    # if os.environ.get("DEV_MODE", "True") == "False" and not token:
    #     raise HTTPException(status_code=401, detail="Authentication required")
        
    logger.info(f"Getting transcription for tabId: {tabId}, videoId: {videoId}")
    
    # Use videoId if provided, otherwise fall back to tabId
    video_id = videoId or tabId
    
    # CRITICAL FIX: Always log the exact video_id we're using
    logger.info(f"EXACT video_id being used: '{video_id}'")
    
    # If this is an upload ID, search by the hash part which is most reliable
    if video_id and video_id.startswith("upload_"):
        # Get the hash part (last segment after last underscore)
        parts = video_id.split("_")
        if len(parts) >= 3:
            hash_part = parts[2]
            logger.info(f"Extracted hash part: '{hash_part}' from video_id")
            
            # Look for files that have this hash but might have different timestamp formats
            transcription_paths = []
            
            # First try the exact file path
            exact_path = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
            transcription_paths.append(exact_path)
            
            # Look for any files containing this hash
            try:
                for file in os.listdir(settings.TRANSCRIPTION_DIR):
                    if hash_part in file and file.endswith(".json"):
                        file_path = os.path.join(settings.TRANSCRIPTION_DIR, file)
                        if file_path != exact_path:  # Don't add exact path twice
                            transcription_paths.append(file_path)
                            logger.info(f"Found potential match by hash: {file}")
            except Exception as e:
                logger.error(f"Error searching for files by hash: {str(e)}")
        else:
            transcription_paths = [
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json")
            ]
    # Check for audio IDs as well
    elif video_id and video_id.startswith("audio_"):
        # Get the hash part (last segment after last underscore)
        parts = video_id.split("_")
        if len(parts) >= 3:
            hash_part = parts[2]
            logger.info(f"Extracted hash part: '{hash_part}' from audio_id")
            
            # Similar lookup strategy as for upload IDs
            transcription_paths = []
            
            # First try the exact file path
            exact_path = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
            transcription_paths.append(exact_path)
            
            # Look for any files containing this hash
            try:
                for file in os.listdir(settings.TRANSCRIPTION_DIR):
                    if hash_part in file and file.endswith(".json"):
                        file_path = os.path.join(settings.TRANSCRIPTION_DIR, file)
                        if file_path != exact_path:  # Don't add exact path twice
                            transcription_paths.append(file_path)
                            logger.info(f"Found potential match by hash: {file}")
            except Exception as e:
                logger.error(f"Error searching for files by hash: {str(e)}")
        else:
            transcription_paths = [
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json")
            ]
    # Handle file_* IDs properly (fix for the issue you're seeing)
    elif video_id and video_id.startswith("file_"):
        # Get the hash part (last segment after last underscore)
        parts = video_id.split("_")
        if len(parts) >= 3:
            hash_part = parts[2]
            logger.info(f"Extracted hash part: '{hash_part}' from file_id")
            
            # Similar lookup strategy as for upload IDs
            transcription_paths = []
            
            # First try the exact file path
            exact_path = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
            transcription_paths.append(exact_path)
            
            # Look for any files containing this hash
            try:
                for file in os.listdir(settings.TRANSCRIPTION_DIR):
                    if hash_part in file and file.endswith(".json"):
                        file_path = os.path.join(settings.TRANSCRIPTION_DIR, file)
                        if file_path != exact_path:  # Don't add exact path twice
                            transcription_paths.append(file_path)
                            logger.info(f"Found potential match by hash: {file}")
            except Exception as e:
                logger.error(f"Error searching for files by hash: {str(e)}")
        else:
            transcription_paths = [
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json")
            ]
    else:
        # Standard path checking for non-upload IDs
        transcription_paths = [
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json"),
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json")
        ]
    
    # Check each possible path
    for path in transcription_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded transcription from: {path}")
                    return {"transcription": data.get("text", "Transcription loaded from file.")}
            except Exception as e:
                logger.error(f"Error loading transcription file {path}: {e}")
    
    # If no file found, try database lookup with improved search
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        with get_db_context() as db:
            # First try exact match
            result = db.execute(
                text("SELECT transcription FROM videos WHERE id::text = :id"),
                {"id": str(video_id)}
            )
            video_row = result.fetchone()
            if video_row and video_row[0]:
                logger.info(f"Found transcription in database with exact match")
                return {"transcription": video_row[0]}
                
            # If no exact match and this is an upload ID or audio ID, try matching by hash
            if video_id and ("upload_" in video_id or "audio_" in video_id or "file_" in video_id) and "_" in video_id:
                hash_part = video_id.split("_")[-1]
                logger.info(f"Trying database lookup by hash part: {hash_part}")
                
                like_result = db.execute(
                    text("SELECT id, transcription FROM videos WHERE id::text LIKE :pattern"),
                    {"pattern": f"%{hash_part}%"}
                )
                like_row = like_result.fetchone()
                
                if like_row and like_row[1]:
                    logger.info(f"Found transcription in database by hash part, ID: {like_row[0]}")
                    return {"transcription": like_row[1]}
    except Exception as db_error:
        logger.error(f"Database error: {str(db_error)}")
    
    # CRITICAL FIX: For error messages, make sure we include the FULL ID
    error_msg = f"No transcription found for this file (ID: {video_id}). You may need to process it first."
    logger.warning(f"Returning error message with FULL video_id: {video_id}")
    
    return {
        "transcription": error_msg
    }

# Enhanced visual analysis function
async def start_visual_analysis(file_path: str, video_id: str, tab_id: str):
    """
    Start visual analysis for a video file with improved error handling
    """
    try:
        logger.info(f"Starting visual analysis for video_id: {video_id}")
        
        # Check for existing visual analysis first
        visual_data_file = os.path.join(settings.TRANSCRIPTION_DIR, f"visual_data_{video_id}.json")
        frames_dir = os.path.join(settings.FRAME_DIR, video_id)
        
        # Send initial status update
        try:
            from app.api.websockets import sio
            await sio.emit('visual_analysis_status', {
                'status': 'started',
                'message': 'Starting visual analysis...',
                'video_id': video_id
            }, room=tab_id)
        except Exception as e:
            logger.error(f"Error sending visual analysis start notification: {str(e)}")
        
        # Get visual analysis service
        visual_service = get_visual_analysis_service()
        
        if not visual_service:
            logger.error("Visual analysis service not available")
            try:
                from app.api.websockets import sio
                await sio.emit('visual_analysis_status', {
                    'status': 'error',
                    'message': 'Visual analysis service not available',
                    'video_id': video_id
                }, room=tab_id)
            except Exception as e:
                logger.error(f"Error sending visual service not available notification: {str(e)}")
            return
        
        # Check if this is an audio file - we should skip visual analysis for audio
        file_ext = os.path.splitext(file_path)[1].lower()
        is_audio_file = file_ext in ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
        
        if is_audio_file or video_id.startswith('audio_'):
            logger.info(f"Skipping visual analysis for audio file: {file_path}")
            try:
                from app.api.websockets import sio
                await sio.emit('visual_analysis_status', {
                    'status': 'skipped',
                    'message': 'Visual analysis skipped for audio files',
                    'video_id': video_id
                }, room=tab_id)
            except Exception as e:
                logger.error(f"Error sending visual analysis skipped notification: {str(e)}")
            return
        
        # Notify client that processing has started
        try:
            from app.api.websockets import sio
            await sio.emit('visual_analysis_status', {
                'status': 'processing',
                'message': 'Processing video for visual analysis...',
                'video_id': video_id
            }, room=tab_id)
        except Exception as e:
            logger.error(f"Error sending visual processing notification: {str(e)}")
        
        # Process the video
        visual_data = await visual_service.process_video(file_path, video_id)
        
        # Update database status
        try:
            from sqlalchemy import text
            from app.utils.database import get_db_context
            with get_db_context() as db:
                # Update video record
                db.execute(
                    text("""
                    UPDATE videos 
                    SET has_visual_analysis = TRUE, 
                        visual_analysis_status = 'completed',
                        visual_analysis_completed_at = CURRENT_TIMESTAMP
                    WHERE id::text = :id
                    """),
                    {"id": str(video_id)}
                )
                db.commit()
                logger.info(f"Updated database with visual analysis completion for video_id: {video_id}")
        except Exception as db_error:
            logger.error(f"Database error updating visual analysis status: {str(db_error)}")
        
        # After processing is complete, retrieve key frames, scenes, and other data
        # to send to the client
        try:
            # Collect frames data
            frames_data = []
            if os.path.exists(frames_dir):
                frames_data_file = os.path.join(frames_dir, "frames_data.json")
                if os.path.exists(frames_data_file):
                    with open(frames_data_file, 'r') as f:
                        frames_data = json.load(f)
                        
            # Collect scenes data
            scenes_data = []
            if isinstance(visual_data, dict) and 'scenes' in visual_data:
                scenes_data = visual_data['scenes']
            
            # Collect highlights data
            highlights_data = []
            if isinstance(visual_data, dict) and 'highlights' in visual_data:
                highlights_data = visual_data['highlights']
                
            from app.api.websockets import sio
            
            # Send completion notification
            await sio.emit('visual_analysis_status', {
                'status': 'completed',
                'message': 'Visual analysis completed',
                'video_id': video_id
            }, room=tab_id)
            
            # Send frames data
            await sio.emit('frame_data', {
                'status': 'success',
                'frames': frames_data[:50],  # Limit to first 50 frames to avoid large payload
                'video_id': video_id
            }, room=tab_id)
            
            # Send scenes data
            if scenes_data:
                await sio.emit('video_scenes', {
                    'status': 'success',
                    'scenes': scenes_data,
                    'video_id': video_id
                }, room=tab_id)
            
            # Send highlights data
            if highlights_data:
                await sio.emit('video_highlights', {
                    'status': 'success',
                    'highlights': highlights_data,
                    'video_id': video_id
                }, room=tab_id)
                
        except Exception as e:
            logger.error(f"Error sending visual results to client: {str(e)}")
        
        logger.info(f"Completed visual analysis for video_id: {video_id}")
        
    except Exception as e:
        logger.error(f"Error in visual analysis: {str(e)}", exc_info=True)
        
        # Notify client of error
        try:
            from app.api.websockets import sio
            await sio.emit('visual_analysis_status', {
                'status': 'error',
                'message': f"Visual analysis failed: {str(e)}",
                'video_id': video_id
            }, room=tab_id)
        except Exception as emit_error:
            logger.error(f"Error sending visual analysis error notification: {str(emit_error)}")

# Helper function to standardize video_id handling
def standardize_video_id(video_id):
    """
    Ensure video_id follows the expected format to prevent inconsistencies
    """
    if not video_id:
        return None
        
    # Fix common patterns that might cause inconsistencies
    if video_id.startswith("upload_"):
        # Common pattern: upload_TIMESTAMP_HASH
        parts = video_id.split("_")
        if len(parts) >= 3:
            # Make sure timestamp isn't truncated (should be ~10 digits)
            timestamp = parts[1]
            hash_part = parts[2]
            
            # If timestamp is truncated, log warning
            if len(timestamp) < 10 and timestamp.isdigit():
                logger.warning(f"Timestamp in video_id appears truncated: {video_id}")
                # We can't reliably fix truncated timestamps, so return as-is
            
            return f"upload_{timestamp}_{hash_part}"
    elif video_id.startswith("audio_"):
        # Same format for audio IDs
        parts = video_id.split("_")
        if len(parts) >= 3:
            timestamp = parts[1]
            hash_part = parts[2]
            
            if len(timestamp) < 10 and timestamp.isdigit():
                logger.warning(f"Timestamp in audio_id appears truncated: {video_id}")
            
            return f"audio_{timestamp}_{hash_part}"
    elif video_id.startswith("file_"):
        # Same format for file IDs
        parts = video_id.split("_")
        if len(parts) >= 3:
            timestamp = parts[1]
            hash_part = parts[2]
            
            if len(timestamp) < 10 and timestamp.isdigit():
                logger.warning(f"Timestamp in file_id appears truncated: {video_id}")
            
            return f"file_{timestamp}_{hash_part}"
    
    return video_id

# API endpoints for the new unified routes
@app.post("/api/v1/videos/upload")
async def api_upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tab_id: str = Form(...),
    video_id: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    """API endpoint for video upload, following the API v1 structure"""
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    # This endpoint simply forwards to the main upload_video function
    return await upload_video(background_tasks, file, tab_id, video_id)

@app.post("/api/v1/audios/upload")
async def api_upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tab_id: str = Form(...),
    audio_id: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    """API endpoint for audio upload, following the API v1 structure"""
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    # This endpoint simply forwards to the audio upload function
    return await upload_audio(background_tasks, file, tab_id, audio_id)

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint for API health check
    """
    return {"message": "Welcome to Luna AI API", "version": "2.0.0"}

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Luna AI"}

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {"status": "ok", "service": "Luna AI"}

@app.get("/api/v1/health")
async def api_v1_health_check():
    """
    Health check endpoint for monitoring
    """
    return {"status": "ok", "service": "Luna AI"}

# ADDED: Transcription endpoint for tab_id lookup (fix for 404 error)
@app.get("/api/transcription")
async def api_transcription_by_tab(tab_id: str = Query(None)):
    """API endpoint for getting transcription by tab ID"""
    if not tab_id:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing tab_id parameter"}
        )
    
    return await get_transcription(tab_id)

# ADDED: Timestamp endpoints that the frontend tries to access
@app.get("/api/v1/videos/{video_id}/timestamps")
@app.get("/api/videos/{video_id}/timestamps")
async def direct_video_timestamps(video_id: str, request: Request):
    """Direct endpoint for getting video timestamps"""
    try:
        # Read timestamps from file directly
        import os
        import json
        
        timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
        if os.path.exists(timestamp_path):
            with open(timestamp_path, "r") as f:
                timestamps_data = json.load(f)
                formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
                
                # Convert to API response format
                response_timestamps = []
                for ts in formatted_timestamps:
                    response_timestamps.append({
                        "time": ts.get("start_time", 0),
                        "timestamp": ts.get("start_time", 0),
                        "text": ts.get("text", ""),
                        "display_time": ts.get("display_time", "00:00"),
                        "end_time": ts.get("end_time", 0)
                    })
                
                return {
                    "status": "success",
                    "video_id": video_id,
                    "timestamps": response_timestamps
                }
        
        # If no file found, return empty list
        return {
            "status": "success",
            "video_id": video_id,
            "timestamps": []
        }
    except Exception as e:
        logger.error(f"Error getting timestamps: {str(e)}")
        return {
            "status": "error",
            "video_id": video_id,
            "message": str(e),
            "timestamps": []
        }

@app.get("/api/timestamps")
async def direct_timestamps_query(
    videoId: str = Query(None),
    request: Request = None
):
    """Direct endpoint for getting timestamps by query"""
    if not videoId:
        return {
            "status": "error",
            "message": "Missing videoId parameter",
            "timestamps": []
        }
    
    # Forward to the endpoint with video_id in path
    return await direct_video_timestamps(videoId, request)

# ADDED: Visual data endpoints that the frontend tries to access
@app.get("/api/v1/videos/{video_id}/visual-data")
@app.get("/api/videos/{video_id}/visual")
@app.get("/api/analyze-visual")
@app.get("/api/v1/visual-analysis")
@app.get("/api/v1/visual-analysis/{video_id}")
@app.get("/api/visual-analysis/{video_id}")
@app.get("/api/v1/visual-analysis")
async def get_visual_data(
    video_id: str = None, 
    videoId: str = Query(None),
    request: Request = None
):
    """Unified endpoint for getting visual data that matches all patterns tried by frontend"""
    # Use path parameter if available, otherwise use query parameter
    video_id = video_id or videoId
    
    if not video_id:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing video_id parameter"}
        )
    
    try:
        # Check for visual data file
        visual_data_path = os.path.join(settings.TRANSCRIPTION_DIR, f"visual_data_{video_id}.json")
        
        if os.path.exists(visual_data_path):
            with open(visual_data_path, "r") as f:
                visual_data = json.load(f)
                return {
                    "status": "success",
                    "video_id": video_id,
                    "visual_data": visual_data
                }
        
        # Check for frames data
        frames_dir = os.path.join(settings.FRAME_DIR, video_id)
        if os.path.exists(frames_dir):
            frames_data_path = os.path.join(frames_dir, "frames_data.json")
            if os.path.exists(frames_data_path):
                with open(frames_data_path, "r") as f:
                    frames_data = json.load(f)
                    return {
                        "status": "success",
                        "video_id": video_id,
                        "frames": frames_data
                    }
        
        # If no data found, return empty result
        return {
            "status": "success",
            "video_id": video_id,
            "visual_data": {},
            "frames": [],
            "message": "No visual data available for this video"
        }
    except Exception as e:
        logger.error(f"Error getting visual data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "video_id": video_id,
                "message": f"Error retrieving visual data: {str(e)}"
            }
        )

# ADDED: POST endpoints for triggering visual analysis
@app.post("/api/v1/videos/{video_id}/analyze-visual")
@app.post("/api/visual-analysis/{video_id}")
@app.post("/api/v1/visual-analysis/{video_id}")
@app.post("/api/analyze-visual")
@app.post("/api/v1/visual-analysis")
async def trigger_visual_analysis_api(
    video_id: str = None,
    videoId: str = Form(None),
    request: Request = None,
    token: str = Depends(oauth2_scheme)
):
    """Unified POST endpoint for triggering visual analysis"""
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    # Use path parameter if available, otherwise use form parameter
    video_id = video_id or videoId
    
    if not video_id:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing video_id parameter"}
        )
    
    # Forward to the proper endpoint
    return await trigger_visual_analysis(video_id, request)

# Debug endpoint to help diagnose ID issues - Disabled in production
@app.get("/debug-video-id")
async def debug_video_id(videoId: str, token: str = Depends(oauth2_scheme)):
    """
    Debug endpoint to check all possible transcription files for a given video ID
    """
    # Check for production - require admin token
    if os.environ.get("PRODUCTION", "False").lower() == "true":
        if not token or "admin" not in token:
            raise HTTPException(status_code=403, detail="Admin access required")
            
    results = {
        "video_id": videoId,
        "possible_paths": [],
        "found_files": [],
        "database_check": None
    }
    
    # Generate all possible pattern combinations
    possible_patterns = [
        f"transcription_{videoId}.json",
        f"transcription{videoId}.json",
        f"transcription-{videoId}.json"
    ]
    
    # If it's an upload ID, try some variations
    if videoId.startswith("upload_"):
        parts = videoId.split("_")
        if len(parts) >= 3:
            timestamp = parts[1]
            hash_part = parts[2]
            possible_patterns.extend([
                f"transcription_upload_{timestamp}_{hash_part}.json",
                f"transcription_upload_{hash_part}.json",
                f"transcription_upload_{timestamp[:4]}_{hash_part}.json",
            ])
    # Also check for audio IDs
    elif videoId.startswith("audio_"):
        parts = videoId.split("_")
        if len(parts) >= 3:
            timestamp = parts[1]
            hash_part = parts[2]
            possible_patterns.extend([
                f"transcription_audio_{timestamp}_{hash_part}.json",
                f"transcription_audio_{hash_part}.json",
            ])
    # Also check for file IDs
    elif videoId.startswith("file_"):
        parts = videoId.split("_")
        if len(parts) >= 3:
            timestamp = parts[1]
            hash_part = parts[2]
            possible_patterns.extend([
                f"transcription_file_{timestamp}_{hash_part}.json",
                f"transcription_file_{hash_part}.json",
            ])
    
    # Check each pattern
    for pattern in possible_patterns:
        path = os.path.join(settings.TRANSCRIPTION_DIR, pattern)
        results["possible_paths"].append(path)
        
        if os.path.exists(path):
            try:
                stats = os.stat(path)
                with open(path, "r") as f:
                    data = json.load(f)
                
                results["found_files"].append({
                    "path": path,
                    "size": stats.st_size,
                    "modified": stats.st_mtime,
                    "contains_video_id": videoId in str(data) if data else False,
                    "video_id_in_data": data.get("video_id") if isinstance(data, dict) else None
                })
            except Exception as e:
                results["found_files"].append({
                    "path": path,
                    "error": str(e)
                })
    
    # Check database
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        with get_db_context() as db:
            # Try exact match
            result = db.execute(
                text("SELECT id, transcription, has_visual_analysis FROM videos WHERE id::text = :id"),
                {"id": str(videoId)}
            )
            row = result.fetchone()
            
            if row:
                results["database_check"] = {
                    "found": True,
                    "id": str(row[0]),
                    "has_transcription": bool(row[1]),
                    "has_visual_analysis": bool(row[2]) if len(row) > 2 else False
                }
            else:
                # Try LIKE search
                if videoId.startswith("upload_") or videoId.startswith("audio_") or videoId.startswith("file_"):
                    parts = videoId.split("_")
                    if len(parts) >= 3:
                        hash_part = parts[2]
                        like_result = db.execute(
                            text("SELECT id, transcription, has_visual_analysis FROM videos WHERE id::text LIKE :pattern"),
                            {"pattern": f"%{hash_part}%"}
                        )
                        like_rows = like_result.fetchall()
                        
                        if like_rows:
                            results["database_check"] = {
                                "found": False,
                                "like_matches": [{
                                    "id": str(r[0]), 
                                    "has_transcription": bool(r[1]),
                                    "has_visual_analysis": bool(r[2]) if len(r) > 2 else False
                                } for r in like_rows]
                            }
                        else:
                            results["database_check"] = {"found": False}
                else:
                    results["database_check"] = {"found": False}
    except Exception as db_error:
        results["database_check"] = {"error": str(db_error)}
    
    # List all files in the transcription directory that might be related
    try:
        all_files = os.listdir(settings.TRANSCRIPTION_DIR)
        # Filter for potentially related files
        related_files = [f for f in all_files if videoId in f]
        results["related_files"] = related_files
    except Exception as e:
        results["file_list_error"] = str(e)
    
    return results

# Visual analysis debug endpoint
@app.get("/debug-visual-analysis/{video_id}")
async def debug_visual_analysis(video_id: str, token: str = Depends(oauth2_scheme)):
    """
    Debug endpoint to check visual analysis for a video
    """
    # Check for production - require admin token
    if os.environ.get("PRODUCTION", "False").lower() == "true":
        if not token or "admin" not in token:
            raise HTTPException(status_code=403, detail="Admin access required")
            
    results = {
        "video_id": video_id,
        "visual_data_paths": [],
        "found_files": [],
        "database_status": None
    }
    
    # Generate possible file paths
    visual_data_paths = [
        os.path.join(settings.TRANSCRIPTION_DIR, f"visual_data_{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"visual_{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"embeddings_{video_id}.json")
    ]
    
    results["visual_data_paths"] = visual_data_paths
    
    # Check each path
    for path in visual_data_paths:
        if os.path.exists(path):
            try:
                stats = os.stat(path)
                results["found_files"].append({
                    "path": path,
                    "size": stats.st_size,
                    "modified": stats.st_mtime
                })
            except Exception as e:
                results["found_files"].append({
                    "path": path,
                    "error": str(e)
                })
    
    # Check database status
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        with get_db_context() as db:
            result = db.execute(
                text("""
                SELECT has_visual_analysis, visual_analysis_status, visual_analysis_completed_at
                FROM videos
                WHERE id::text = :id
                """),
                {"id": str(video_id)}
            )
            row = result.fetchone()
            
            if row:
                results["database_status"] = {
                    "has_visual_analysis": bool(row[0]) if row[0] is not None else False,
                    "visual_analysis_status": row[1],
                    "completed_at": str(row[2]) if row[2] else None
                }
            else:
                results["database_status"] = {"found": False}
    except Exception as db_error:
        results["database_status"] = {"error": str(db_error)}
    
    # List frames directory
    frames_dir = os.path.join(settings.FRAME_DIR, video_id)
    if os.path.exists(frames_dir):
        try:
            frame_files = os.listdir(frames_dir)
            results["frames_count"] = len(frame_files)
            results["frames_examples"] = frame_files[:5] if len(frame_files) > 0 else []
        except Exception as e:
            results["frames_error"] = str(e)
    else:
        results["frames_dir"] = f"Directory not found: {frames_dir}"
    
    return results

# Endpoint to manually trigger visual analysis
@app.post("/api/v1/videos/{video_id}/trigger-visual-analysis")
async def trigger_visual_analysis(
    video_id: str, 
    request: Request,
    token: str = Depends(oauth2_scheme)
):
    """
    Manually trigger visual analysis for a video
    """
    # Validate token in production mode
    if os.environ.get("DEV_MODE", "True") == "False" and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    logger.info(f"Manual trigger of visual analysis for video_id: {video_id}")
    
    try:
        # Look up video file path
        from sqlalchemy import text
        from app.utils.database import get_db_context
        
        file_path = None
        
        # First check if video exists in database
        with get_db_context() as db:
            result = db.execute(
                text("SELECT video_url FROM videos WHERE id::text = :id"),
                {"id": str(video_id)}
            )
            row = result.fetchone()
            
            if row and row[0]:
                file_path = row[0]
            
        # If not found in database, try to find the file in the uploads directory
        if not file_path:
            upload_dir = Path(settings.UPLOAD_DIR)
            # Try common extensions
            for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv']:
                potential_path = os.path.join(upload_dir, f"{video_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        
        if not file_path:
            return JSONResponse(
                status_code=404, 
                content={"error": f"Video file not found for ID: {video_id}"}
            )
        
        # Get client IP or connection info for tab_id
        client_host = request.client.host if request.client else "unknown"
        tab_id = video_id  # Use video_id as tab_id for manual triggers
        
        # Start visual analysis in background
        asyncio.create_task(start_visual_analysis(file_path, video_id, tab_id))
        
        return {
            "status": "success",
            "message": "Visual analysis started",
            "video_id": video_id
        }
    
    except Exception as e:
        logger.error(f"Error triggering visual analysis: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={"error": f"Failed to trigger visual analysis: {str(e)}"}
        )

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {str(exc)}"},
    )
    
# Authentication endpoints for login and registration
@app.post("/api/auth/login")
async def direct_login(request: Request):
    """Authentication endpoint for login"""
    try:
        body = await request.json()
        email = body.get("email", "")
        password = body.get("password", "")
        
        # Log login attempt
        logger.info(f"Login attempt received for email: {email}")
        
        # Return successful response with redirectUrl to analysis page
        return {
            "status": "success",
            "token": f"token_{int(time.time())}",
            "user": {
                "id": f"user_{uuid.uuid4()}",
                "email": email,
                "name": email.split('@')[0]
            },
            "redirectUrl": "/analysis"  # Add this to redirect to analysis page
        }
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/api/auth/register")
async def direct_register(request: Request):
    """Authentication endpoint for registration"""
    try:
        body = await request.json()
        email = body.get("email", "")
        password = body.get("password", "")
        name = body.get("name", email.split('@')[0] if '@' in email else "New User")
        
        logger.info(f"Registration attempt received for email: {email}")
        
        # Return successful response with redirectUrl to analysis page
        return {
            "status": "success",
            "token": f"token_{int(time.time())}",
            "user": {
                "id": f"user_{uuid.uuid4()}",
                "email": email,
                "name": name
            },
            "redirectUrl": "/analysis"  # Add this to redirect to analysis page
        }
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Token refresh endpoint
@app.post("/api/auth/refresh")
async def refresh_token(request: Request):
    """Refresh an authentication token"""
    try:
        # Get authorization header
        auth_header = request.headers.get("Authorization", "")
        
        # Extract token
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        
        if not token:
            raise HTTPException(status_code=401, detail="Invalid or missing token")
            
        # In a real implementation, validate the token
        # For now, just return a new token
        return {
            "status": "success",
            "token": f"refreshed_token_{int(time.time())}",
            "expiresIn": 3600  # Token expires in 1 hour
        }
    except HTTPException as http_error:
        # Re-raise HTTP exceptions
        raise http_error
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Signout endpoint
@app.post("/api/auth/logout")
async def logout(request: Request):
    """End a user session"""
    # In a real implementation, add the token to a blacklist
    return {"status": "success", "message": "Logged out successfully"}

# Start the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload in production
        workers=min(4, os.cpu_count() or 1)  # Set workers based on CPU cores
    )