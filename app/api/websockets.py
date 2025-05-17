"""
Enhanced WebSockets API for Luna AI using Socket.IO
Handles real-time communication for chat and video processing status
with improved NLU, contextual Q&A, and topic analysis
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Response
import json
import os
import re
import logging
import traceback
import asyncio
import tempfile
import subprocess
import uuid
import time
import random  # Added for ID generation
from typing import Dict, Any, Optional, List
from pathlib import Path
# Import python-socketio
import socketio
# Import from our simplified structure
from app.utils.database import get_db
from app.config import settings
from app.services.transcription import (
    transcribe_video as transcribe_video_service, 
    get_timestamps_for_video, 
    transcribe_youtube_video as transcribe_youtube_video_service  # Renamed to avoid conflict
)
from app.services.transcription_service import (
    process_youtube_video, get_task_status, cache
)

# Environment variables for configuration
DEBUG_TRANSCRIPTION = os.environ.get("DEBUG_TRANSCRIPTION") == "1"
TRANSCRIPTION_SERVICE = os.environ.get("TRANSCRIPTION_SERVICE", "google")  # Options: google, assembly, whisper
USE_PROXY = os.environ.get("USE_PROXY", "0") == "1"
PROXY_URL = os.environ.get("PROXY_URL", "")

# Try to import visual analysis service if available
try:
    from app.services.visual_analysis import get_visual_analysis_service, get_visual_data, analyze_visual_data, get_visual_analysis_data
except ImportError:
    # Define minimal stand-ins for the visual services if not available
    def get_visual_analysis_service():
        return None
    def get_visual_data(db, video_id):
        return {"status": "pending", "message": "Visual analysis not available", "video_id": video_id}
    def analyze_visual_data(db, video_id):
        return {"status": "pending", "message": "Visual analysis not available", "video_id": video_id}
    def get_visual_analysis_data(db, video_id):
        return {"status": "pending", "message": "Visual analysis not available", "video_id": video_id}

router = APIRouter()
# Configure logging
logger = logging.getLogger("websockets")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# Set up file handler if log directory exists
log_dir = os.environ.get('LOG_DIR', '/tmp')
if os.path.exists(log_dir):
    file_handler = logging.FileHandler(os.path.join(log_dir, 'websockets.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
# Explicitly define allowed origins - CRITICAL FIX
ALLOWED_ORIGINS = [
    "http://localhost:5173",     # Development frontend
    "http://127.0.0.1:5173",     # Alternative localhost
    "http://localhost:8000",     # Backend URL
    "http://127.0.0.1:8000",     # Alternative backend URL
    "*"                          # Allow all origins in development - REMOVE IN PRODUCTION
]
logger.info(f"Socket.IO allowed origins: {ALLOWED_ORIGINS}")
# Create a Socket.IO server with explicit version compatibility settings
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=ALLOWED_ORIGINS,
    logger=True,
    engineio_logger=True,
    json=json,
)
# Set explicit EIO settings - INCREASED TIMEOUTS TO FIX CONNECTION ISSUES
sio.eio.max_http_buffer_size = 50000000  # 50MB for larger payloads - INCREASED from 10MB
sio.eio.ping_timeout = 1200   # Increased timeout to 20 minutes - DOUBLED from 10 minutes
sio.eio.ping_interval = 30    # Decreased to 30 seconds for more frequent pings

# CHANGED: Disable debug mode by default
# Only use DEBUG_TRANSCRIPTION if it's explicitly set in the environment
debug_mode_enabled = DEBUG_TRANSCRIPTION
if debug_mode_enabled:
    logger.info("⚠️ Debug transcription mode is enabled from environment")
else:
    logger.info("🔴 Using actual transcription services (debug mode off)")

# Create an ASGI app for Socket.IO
socket_app = socketio.ASGIApp(sio)
# Store session data
session_data = {}

# ADDED: Define tab_id to client mapping dictionary
# This allows us to emit messages to specific tabs
tab_to_sid_mapping = {}

# ADDED: Track processed message IDs to prevent duplicate responses
processed_message_ids = set()
# Limit the size of the processed_message_ids set
MAX_PROCESSED_MESSAGES = 1000

# ADDED: Standardize ID formats function
def standardize_video_id(video_id):
    """
    Standardize video ID format to ensure consistency between frontend and backend
    
    Args:
        video_id (str): Input video ID in any format
    
    Returns:
        tuple: (standardized_id, upload_id, file_id)
    """
    if not video_id:
        return None, None, None
        
    # Extract timestamp and hash parts
    upload_match = re.search(r'upload_(\d+)(?:_([a-z0-9]+))?', video_id)
    file_match = re.search(r'file_(\d+)', video_id)
    
    if upload_match:
        timestamp = upload_match.group(1)
        hash_part = upload_match.group(2) if upload_match.group(2) else ""
        
        # Create standardized IDs
        upload_id = f"upload_{timestamp}_{hash_part}" if hash_part else f"upload_{timestamp}"
        file_id = f"file_{timestamp}"
        
        return upload_id, upload_id, file_id
        
    elif file_match:
        timestamp = file_match.group(1)
        
        # Try to find matching upload ID in session data
        upload_id = None
        for sid_data in session_data.values():
            if 'current_video_id' in sid_data and sid_data['current_video_id']:
                current_id = sid_data['current_video_id']
                if f"upload_{timestamp}" in current_id:
                    upload_id = current_id
                    break
        
        # If no match found, use generic upload ID
        if not upload_id:
            upload_id = f"upload_{timestamp}"
            
        file_id = f"file_{timestamp}"
        
        return file_id, upload_id, file_id
        
    # If doesn't match expected formats, return as is
    return video_id, video_id, video_id

def serialize_for_json(obj):
    """
    Ensure object is serializable for JSON
    """
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(i) for i in obj]
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return serialize_for_json(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return serialize_for_json(obj.__dict__)
    else:
        # Convert non-serializable objects to strings
        if not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj

# Ensure the video_timestamps table exists
async def ensure_timestamp_table_exists():
    """Create the video_timestamps table if it doesn't exist"""
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        
        with get_db_context() as db:
            db.execute(text("""
            CREATE TABLE IF NOT EXISTS video_timestamps (
                id SERIAL PRIMARY KEY,
                video_id TEXT NOT NULL,
                timestamp FLOAT NOT NULL,
                formatted_time TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """))
            db.commit()
            logger.info("Ensured video_timestamps table exists")
    except Exception as e:
        logger.error(f"Error creating timestamps table: {e}")

# Run the table creation task on startup
try:
    asyncio.create_task(ensure_timestamp_table_exists())
except Exception as e:
    logger.error(f"Failed to create timestamp table task: {e}")

def setup_fastapi_routes(app):
    """
    Add REST API endpoints for timestamps to prevent 404 errors
    These routes will complement the Socket.IO handlers
    """
    # Add timestamps API endpoints
    @app.get("/api/timestamps/{video_id}")
    async def get_timestamps_api(video_id: str, request: Request):
        """REST API endpoint for getting timestamps"""
        logger.info(f"REST API request for timestamps: {video_id}")
        try:
            timestamps_data = await get_timestamps_for_video(video_id)
            if not timestamps_data or not timestamps_data.get("formatted_timestamps"):
                # Get transcript and generate fallback timestamps
                transcript_text = await load_transcription(video_id)
                if transcript_text:
                    timestamps_data = await generate_fallback_timestamps(transcript_text, video_id)
                else:
                    return {"status": "error", "message": "No transcript available", "timestamps": []}
            
            return {
                "status": "success",
                "video_id": video_id,
                "timestamps": timestamps_data.get("formatted_timestamps", [])
            }
        except Exception as e:
            logger.error(f"Error in timestamps API: {str(e)}")
            return {"status": "error", "message": str(e), "timestamps": []}
    
    # Add v1 API endpoints
    @app.get("/api/v1/timestamps")
    async def get_timestamps_v1_api(video_id: str = None, request: Request = None):
        """V1 REST API endpoint for getting timestamps with query param"""
        logger.info(f"V1 REST API request for timestamps (query): {video_id}")
        if not video_id:
            # Try to get it from the path
            path = request.url.path
            match = re.search(r'youtube_[^/&?]+', path)
            if match:
                video_id = match.group(0)
                logger.info(f"Extracted video_id from path: {video_id}")
        return await get_timestamps_api(video_id, request)
    
    @app.get("/api/v1/timestamps/{video_id}")
    async def get_timestamps_v1_path_api(video_id: str, request: Request):
        """V1 REST API endpoint for getting timestamps with path param"""
        logger.info(f"V1 REST API request for timestamps (path): {video_id}")
        return await get_timestamps_api(video_id, request)
    
    # Add video timestamps endpoints
    @app.get("/api/v1/timestamps/video/{video_id}")
    async def get_video_timestamps_v1_api(video_id: str, request: Request):
        """V1 REST API endpoint for video timestamps"""
        logger.info(f"V1 REST API request for video timestamps: {video_id}")
        return await get_timestamps_api(video_id, request)
        
    @app.get("/api/timestamps/video/{video_id}")
    async def get_video_timestamps_api(video_id: str, request: Request):
        """REST API endpoint for video timestamps"""
        logger.info(f"REST API request for video timestamps: {video_id}")
        return await get_timestamps_api(video_id, request)
    
    # ADDED: Handle transcription API endpoints that were causing 404 errors
    @app.get("/api/v1/videos/{video_id}/transcription")
    async def get_video_transcription_api(video_id: str, request: Request):
        """REST API endpoint for video transcription"""
        logger.info(f"REST API request for video transcription: {video_id}")
        try:
            # Standardize the video ID
            _, upload_id, _ = standardize_video_id(video_id)
            
            # Load the transcription
            transcript = await load_transcription(upload_id)
            
            return {
                "status": "success", 
                "video_id": video_id,
                "transcript": transcript
            }
        except Exception as e:
            logger.error(f"Error in transcription API: {str(e)}")
            return {"status": "error", "message": str(e), "transcript": ""}

    # Add endpoint for file/upload transcriptions
    @app.get("/api/uploads/{upload_id}/transcription")
    async def get_upload_transcription_api(upload_id: str, request: Request):
        """REST API endpoint for uploaded file transcription"""
        logger.info(f"REST API request for upload transcription: {upload_id}")
        return await get_video_transcription_api(upload_id, request)
    
    # Add POST endpoint for transcriptions
    @app.post("/api/v1/transcription")
    async def post_transcription_api(request: Request):
        """POST endpoint for transcription requests"""
        try:
            body = await request.json()
            video_id = body.get("video_id")
            
            if not video_id:
                return {"status": "error", "message": "Missing video_id"}
                
            # Standardize the video ID
            _, upload_id, _ = standardize_video_id(video_id)
            
            # Load the transcription
            transcript = await load_transcription(upload_id)
            
            return {
                "status": "success", 
                "video_id": video_id,
                "transcript": transcript
            }
        except Exception as e:
            logger.error(f"Error in POST transcription API: {str(e)}")
            return {"status": "error", "message": str(e), "transcript": ""}
    
    # Add visual analysis endpoint to prevent 404 errors
    @app.get("/api/visual-analysis/{video_id}")
    async def get_visual_analysis_api(video_id: str, request: Request):
        """REST API endpoint for visual analysis"""
        logger.info(f"REST API request for visual analysis: {video_id}")
        try:
            from app.utils.database import get_db_context
            with get_db_context() as db:
                visual_data = get_visual_data(db, video_id)
                return visual_data
        except Exception as e:
            logger.error(f"Error in visual analysis API: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "video_id": video_id,
                "data": {
                    "frames": [],
                    "scenes": [],
                    "topics": [],
                    "highlights": []
                }
            }
    
    # Add catch-all for visual-analysis URLs
    @app.get("/api/visual-analysis/youtube_{youtube_id}")
    async def get_visual_analysis_youtube_api(youtube_id: str, request: Request):
        """Special catch-all for YouTube visual analysis requests"""
        video_id = f"youtube_{youtube_id}"
        logger.info(f"Special visual analysis API for: {video_id}")
        return await get_visual_analysis_api(video_id, request)
    
    # Log the registered routes
    logger.info("Registered REST API endpoints for timestamps to prevent 404 errors")

def detect_youtube_video_type(url):
    """
    Detect YouTube video type and determine appropriate settings
    
    Returns:
        tuple: (video_type, timeout, priority)
    """
    if not url:
        return "unknown", 300, "normal"
        
    url = url.lower()  # Normalize URL for better detection
    
    if "youtube.com/shorts" in url or "youtu.be" in url and "/shorts/" in url:
        # Short-form content - typically 15-60 seconds
        return "shorts", 600, "high"  # 10-minute timeout (doubled from 300s), high priority
    elif "youtube.com/clip" in url:
        # YouTube clip - usually shorter than full videos
        return "clip", 720, "high"  # 12-minute timeout (doubled from 360s), high priority
    elif "t=" in url or "start=" in url:
        # Regular video with timestamp - likely focusing on a specific section
        return "timestamped", 900, "normal"  # 15-minute timeout (doubled from 420s), normal priority
    else:
        # Regular full YouTube video
        return "regular", 1200, "normal"  # 20-minute timeout (doubled from 600s), normal priority

# COMPLETELY REWRITTEN YouTube download function with enhanced reliability
async def download_youtube_video(youtube_url, output_path, sid=None):
    """
    Try to download a YouTube video with enhanced reliability and fallback mechanisms
    
    Args:
        youtube_url: URL of YouTube video to download
        output_path: Path to save the file
        sid: Socket ID for status updates (optional)
        
    Returns:
        tuple: (success, message, metadata)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # First try to get the video title and duration using yt-dlp with improved options
    metadata = {}
    video_title = None
    video_duration = None
    
    try:
        # Use yt-dlp with enhanced options for metadata
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--print", "title",
            "--print", "duration",
            "--no-check-certificate",
            "--geo-bypass",
            "--geo-bypass-country", "US",
            "--socket-timeout", "60",
        ]
        
        # Add proxy if configured
        if USE_PROXY and PROXY_URL:
            cmd.extend(["--proxy", PROXY_URL])
            
        # Add the URL at the end
        cmd.append(youtube_url)
        
        # Run the command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode().strip().split("\n")
        
        if process.returncode == 0 and len(stdout_text) >= 2:
            video_title = stdout_text[0].strip()
            try:
                video_duration = int(stdout_text[1].strip())
            except ValueError:
                video_duration = 300  # Default 5 minutes
                
            metadata["title"] = video_title
            metadata["duration"] = video_duration
            logger.info(f"Got video metadata: {video_title}, duration: {video_duration}s")
        else:
            stderr_text = stderr.decode().strip()
            logger.warning(f"Could not get video metadata: {stderr_text}")
            video_title = "Unknown YouTube Video"
            video_duration = 300  # Default 5 minutes
            metadata["title"] = video_title
            metadata["duration"] = video_duration
    except Exception as e:
        logger.error(f"Error getting video metadata: {e}")
        video_title = "Unknown YouTube Video"
        video_duration = 300  # Default 5 minutes
        metadata["title"] = video_title
        metadata["duration"] = video_duration
    
    # Enhanced download methods with multiple fallbacks
    methods = [
        {
            "name": "Standard download with geo-bypass",
            "cmd": [
                "yt-dlp",
                "-f", "best[height<=720]/best",
                "--max-filesize", "200M",  # Increased from 100M
                "--socket-timeout", "60",  # Increased from 30
                "--no-check-certificate",
                "--geo-bypass",
                "--geo-bypass-country", "US",
                "-o", output_path,
            ]
        },
        {
            "name": "Audio only with cookies",
            "cmd": [
                "yt-dlp",
                "-f", "bestaudio",
                "--extract-audio",
                "--socket-timeout", "60",
                "--no-check-certificate",
                "--geo-bypass",
                "-o", output_path,
            ]
        },
        {
            "name": "Low quality reliable fallback",
            "cmd": [
                "yt-dlp",
                "-f", "worst",
                "--no-check-certificate",
                "--geo-bypass",
                "--no-playlist",
                "--socket-timeout", "60",
                "-o", output_path,
            ]
        },
        {
            "name": "Last resort with format override",
            "cmd": [
                "yt-dlp",
                "--no-check-certificate",
                "--geo-bypass",
                "--force-generic-extractor",
                "--socket-timeout", "90",
                "--default-search", "ytsearch:",
                "-o", output_path,
            ]
        }
    ]
    
    # Try each method with improved proxy handling
    for method in methods:
        method_name = method["name"]
        cmd = method["cmd"].copy()  # Create a copy to avoid modifying the original
        
        # Add proxy if configured
        if USE_PROXY and PROXY_URL:
            cmd.extend(["--proxy", PROXY_URL])
            
        # Add the URL at the end if not already last-resort method
        if "Last resort" not in method_name:
            cmd.append(youtube_url)
        else:
            # For last resort, add search term
            search_term = youtube_url
            if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
                # Extract video ID
                video_id = None
                if "youtube.com/watch?v=" in youtube_url:
                    video_id = youtube_url.split("watch?v=")[1].split("&")[0]
                elif "youtu.be/" in youtube_url:
                    video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
                
                if video_id:
                    search_term = f"ytsearch:{video_title or video_id}"
            
            cmd.append(search_term)
        
        try:
            logger.info(f"Trying YouTube download with method: {method_name}")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stderr_text = stderr.decode().strip()
            
            if process.returncode == 0:
                logger.info(f"YouTube download successful with method: {method_name}")
                
                # Verify file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # Log success details
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                    logger.info(f"Download complete - File size: {file_size:.2f} MB")
                    return True, "Download successful", {
                        "method": method_name,
                        "title": video_title,
                        "duration": video_duration,
                        "file_size_mb": file_size
                    }
                else:
                    logger.warning(f"Download succeeded but file missing or empty: {output_path}")
            else:
                logger.warning(f"Download failed with method {method_name}: {stderr_text}")
                # Add detailed error logging
                if "This video is unavailable" in stderr_text:
                    logger.error("YouTube reports video is unavailable")
                elif "Unable to extract" in stderr_text:
                    logger.error("YouTube extraction failed - possible site changes")
                elif "HTTP Error" in stderr_text:
                    logger.error(f"HTTP error detected: {stderr_text}")
        except Exception as e:
            logger.error(f"Error with method {method_name}: {e}", exc_info=True)
    
    # If all methods fail, try direct API fetch if environment variables set
    if os.environ.get("YT_API_KEY"):
        try:
            logger.info("Attempting YouTube API fetch as last resort")
            # Implementation would go here if YouTube API keys are available
            # This is a stub for future implementation
        except Exception as api_error:
            logger.error(f"YouTube API fetch failed: {api_error}")
    
    # Create better synthetic placeholder if all download methods fail
    try:
        # Create a minimal valid media file as placeholder
        with open(output_path, 'wb') as f:
            # Write a valid MP4 header and minimal data
            f.write(b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41\x00\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00' * 16)  # Add some padding
        
        logger.warning(f"Created MP4 placeholder file at {output_path}")
        
        # Better metadata for fallback
        return False, "Failed to download video but created placeholder", {
            "title": video_title,
            "duration": video_duration,
            "error": "YouTube extraction failed - possible restrictions or regional blocks",
            "fallback": True
        }
    except Exception as e:
        logger.error(f"Failed to create placeholder file: {e}")
        return False, f"Error: {e}", {"fallback": True}

# Generate a synthetic transcription for when download fails
def create_synthetic_transcription(video_id, url, metadata=None):
    """
    Create a synthetic transcription when YouTube download fails
    
    Args:
        video_id: Video ID
        url: YouTube URL
        metadata: Video metadata (title, duration, etc.)
        
    Returns:
        dict: Transcription data structure
    """
    if not metadata:
        metadata = {}
    
    title = metadata.get("title", "Unknown YouTube Video")
    duration = metadata.get("duration", 300)
    error = metadata.get("error", "HTTP Error 400: Bad Request")
    
    # Create a sensible looking synthetic transcript
    text = f"[Unable to transcribe video due to download error: {error}]\n\n"
    text += f"This video appears to be titled '{title}' with approximately {duration} seconds duration.\n"
    text += f"Video ID: {video_id}\nURL: {url}\n\n"
    text += "The video content could not be accessed due to YouTube restrictions or connection issues."
    
    # Create segments for the transcript
    segments = [
        {
            "id": 0,
            "start": 0.0,
            "end": min(30.0, duration),
            "text": f"[Unable to transcribe video due to download error: {error}]"
        },
        {
            "id": 1,
            "start": min(30.0, duration),
            "end": min(60.0, duration),
            "text": f"This video appears to be titled '{title}'."
        },
        {
            "id": 2,
            "start": min(60.0, duration),
            "end": duration,
            "text": "The video content could not be accessed due to YouTube restrictions or connection issues."
        }
    ]
    
    # Generate word-level timestamps
    words = text.split()
    word_count = len(words)
    word_duration = duration / max(1, word_count)
    
    timestamps = []
    for i, word in enumerate(words):
        start_time = i * word_duration
        end_time = (i + 1) * word_duration
        if start_time < duration:
            timestamps.append({
                "word": word,
                "start_time": start_time,
                "end_time": min(end_time, duration)
            })
    
    # Generate formatted timestamps
    formatted_timestamps = []
    seg_count = min(10, max(3, duration // 30))
    seg_duration = duration / seg_count
    
    for i in range(seg_count):
        start_time = i * seg_duration
        end_time = (i + 1) * seg_duration
        
        # Format display time
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        display_time = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
        
        # Create a segment text
        if i == 0:
            segment_text = f"[Unable to transcribe video due to download error: {error}]"
        elif i == 1:
            segment_text = f"This video appears to be titled '{title}'."
        else:
            segment_text = "The video content could not be accessed due to YouTube restrictions."
            
        formatted_timestamps.append({
            "text": segment_text,
            "start_time": start_time,
            "end_time": end_time,
            "display_time": display_time
        })
    
    # Build the complete transcription data structure
    transcription = {
        "text": text,
        "segments": segments,
        "timestamps": timestamps,
        "formatted_timestamps": formatted_timestamps,
        "error": error,
        "synthetic": True,
        "video_id": video_id,
        "metadata": metadata
    }
    
    return transcription

# Clean up processed message IDs to prevent memory leaks
async def cleanup_processed_message_ids():
    """Periodically clean up the processed_message_ids set to prevent memory leaks"""
    global processed_message_ids
    
    if len(processed_message_ids) > MAX_PROCESSED_MESSAGES:
        # Keep only the most recent half of the messages
        processed_message_ids = set(list(processed_message_ids)[-MAX_PROCESSED_MESSAGES//2:])
        logger.info(f"Cleaned up processed_message_ids, now tracking {len(processed_message_ids)} messages")

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    client_id = environ.get('HTTP_X_CLIENT_ID', sid)
    logger.info(f"Client {client_id} connected with SID: {sid}")
    
    # Store session info
    session_data[sid] = {'client_id': client_id}
    
    # Log connection details for debugging
    headers = {k: v for k, v in environ.items() if k.startswith('HTTP_')}
    logger.info(f"Connection headers: {headers}")
    logger.info(f"Connection path: {environ.get('PATH_INFO')}")
    
    # Emit welcome event
    await sio.emit('connection_established', {'status': 'connected', 'sid': sid}, to=sid)
    
    # Debug connection info
    logger.info(f"Connected clients: {len(session_data)}")
    logger.info(f"Connection origin: {environ.get('HTTP_ORIGIN', 'unknown origin')}")
    
    # Start heartbeat for this connection
    asyncio.create_task(send_heartbeat(sid))

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    logger.info(f"Client {client_id} disconnected. SID: {sid}")
    
    # Clean up tab ID mappings
    tab_id = session_data.get(sid, {}).get('tab_id')
    if tab_id and tab_id in tab_to_sid_mapping:
        if tab_to_sid_mapping[tab_id] == sid:
            del tab_to_sid_mapping[tab_id]
            logger.info(f"Removed tab ID mapping for {tab_id}")
    
    if sid in session_data:
        # Clean up active requests
        if 'active_requests' in session_data[sid]:
            session_data[sid]['active_requests'].clear()
        del session_data[sid]

@sio.event
async def error(sid, error_data):
    """Handle Socket.IO errors"""
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    logger.error(f"Socket.IO error for client {client_id}: {error_data}")

@sio.event
async def echo(sid, data):
    """
    Simple echo event handler for testing Socket.IO connection
    """
    logger.info(f"Received echo from {sid}: {data}")
    
    await sio.emit('echo_response', {
        'status': 'success',
        'message': 'Echo received',
        'received_data': data,
        'sid': sid
    }, to=sid)

# ADDED: Helper function to emit to tab ID
async def emit_to_tab(event, data, tab_id):
    """
    Emit an event to a specific tab ID
    
    Args:
        event (str): Event name
        data (dict): Event data
        tab_id (str): Tab ID to emit to
    
    Returns:
        bool: True if emitted successfully, False otherwise
    """
    if not tab_id:
        return False
        
    # Find SID for this tab
    sid = tab_to_sid_mapping.get(tab_id)
    
    if sid:
        try:
            await sio.emit(event, data, to=sid)
            return True
        except Exception as e:
            logger.error(f"Error emitting to tab {tab_id}: {e}")
            return False
    
    # If not found in mapping, try broadcasting to all sessions with this tab_id
    emitted = False
    for curr_sid, session in session_data.items():
        if session.get('tab_id') == tab_id:
            try:
                await sio.emit(event, data, to=curr_sid)
                emitted = True
            except Exception as e:
                logger.error(f"Error broadcasting to tab {tab_id} on SID {curr_sid}: {e}")
    
    return emitted

# FIXED: Function to check if a message has already been processed
def is_message_processed(message_id):
    """Check if a message ID has already been processed to prevent duplicates"""
    return message_id in processed_message_ids

# FIXED: Function to mark a message as processed
def mark_message_processed(message_id):
    """Mark a message ID as processed to prevent duplicates"""
    processed_message_ids.add(message_id)
    # Schedule cleanup if the set is getting too large
    if len(processed_message_ids) > MAX_PROCESSED_MESSAGES:
        asyncio.create_task(cleanup_processed_message_ids())

# FIXED: Completely rewritten ask_ai_with_context function to prevent duplicate responses
@sio.event
async def ask_ai_with_context(sid, data):
    """
    Handle AI questions over Socket.IO with conversation context
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received ask_ai_with_context request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        user_question = data.get('question')
        video_id = data.get('video_id') or data.get('tabId')  # Accept either name
        conversation_history = data.get('conversation_history', [])
        message_id = data.get('messageId') or f"{video_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # CRITICAL FIX: Check for duplicate messages
        if is_message_processed(message_id):
            logger.warning(f"Ignoring duplicate message with ID: {message_id}")
            return
            
        # Mark this message as processed immediately
        mark_message_processed(message_id)
        
        if not user_question or not video_id:
            await sio.emit('error', {'message': 'Missing question or video_id'}, to=sid)
            return
        
        logger.info(f"Processing AI question with context: '{user_question}' for video {video_id}")
        
        # Let the client know we're processing the question
        await sio.emit('ai_thinking', {
            'status': 'thinking',
            'video_id': video_id,
            'question': user_question,
            'message_id': message_id
        }, to=sid)
        
        # Store in session data to track this request
        if sid not in session_data:
            session_data[sid] = {}
        if 'active_requests' not in session_data[sid]:
            session_data[sid]['active_requests'] = set()
        session_data[sid]['active_requests'].add(message_id)
        
        try:
            from app.services.ai import generate_ai_response_with_history
            
            # Generate AI response with conversation history
            response = await generate_ai_response_with_history(
                video_id=video_id,
                question=user_question,
                conversation_history=conversation_history,
                tab_id=sid
            )
            
            # CRITICAL FIX: Check if this request is still active before responding
            # And send only ONE response with completion flag
            if sid in session_data and 'active_requests' in session_data[sid] and message_id in session_data[sid]['active_requests']:
                # Send combined response with completion flag
                await sio.emit('ai_response', {
                    'status': 'success',
                    'answer': response.get('answer', 'I couldn\'t generate a response'),
                    'timestamps': response.get('timestamps', []),
                    'question': user_question,
                    'video_id': video_id,
                    'message_id': message_id,
                    'is_complete': True  # Add completion flag here
                }, to=sid)
                
                # Remove this request from active requests
                session_data[sid]['active_requests'].remove(message_id)
            else:
                logger.info(f"Request {message_id} was cancelled or session expired, not sending response")
        except ImportError:
            logger.error("ai.py module not found or missing generate_ai_response_with_history function")
            # Fall back to standard AI response if enhanced version is not available
            from app.services.ai import generate_ai_response
            response = await generate_ai_response(
                video_id=video_id,
                question=user_question,
                tab_id=sid
            )
            
            # CRITICAL FIX: Check if this request is still active before responding
            # And send only ONE response with completion flag
            if sid in session_data and 'active_requests' in session_data[sid] and message_id in session_data[sid]['active_requests']:
                await sio.emit('ai_response', {
                    'status': 'success',
                    'answer': response.get('answer', 'I couldn\'t generate a response'),
                    'timestamps': response.get('timestamps', []),
                    'question': user_question,
                    'video_id': video_id,
                    'message_id': message_id,
                    'is_complete': True  # Add completion flag here
                }, to=sid)
                
                # Remove this request from active requests
                session_data[sid]['active_requests'].remove(message_id)
            else:
                logger.info(f"Request {message_id} was cancelled or session expired, not sending response")
                
    except Exception as e:
        logger.error(f"Error in ask_ai_with_context: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('error', {'message': f'An error occurred: {str(e)}'}, to=sid)
        
        # Make sure to send error message with completion flag
        await sio.emit('ai_response', {
            'status': 'error',
            'answer': f'An error occurred while processing your request: {str(e)}',
            'video_id': video_id if 'video_id' in locals() else None,
            'message_id': message_id if 'message_id' in locals() else None,
            'is_complete': True
        }, to=sid)

# FIXED: Also update the ask_ai function with the same fix
@sio.event
async def ask_ai(sid, data):
    """
    Legacy handler for AI questions without conversation context
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received legacy ask_ai request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        user_question = data.get('question')
        video_id = data.get('video_id') or data.get('tabId')  # Accept either name
        message_id = data.get('messageId') or f"{video_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # CRITICAL FIX: Check for duplicate messages
        if is_message_processed(message_id):
            logger.warning(f"Ignoring duplicate message with ID: {message_id}")
            return
            
        # Mark this message as processed immediately
        mark_message_processed(message_id)
        
        if not user_question or not video_id:
            await sio.emit('error', {'message': 'Missing question or video_id'}, to=sid)
            return
        
        logger.info(f"Processing legacy AI question: '{user_question}' for video {video_id}")
        
        # Store in session data to track this request
        if sid not in session_data:
            session_data[sid] = {}
        if 'active_requests' not in session_data[sid]:
            session_data[sid]['active_requests'] = set()
        session_data[sid]['active_requests'].add(message_id)
        
        # Let the client know we're processing the question
        await sio.emit('ai_thinking', {
            'status': 'thinking',
            'video_id': video_id,
            'question': user_question,
            'message_id': message_id
        }, to=sid)
        
        # Use the standard AI response function
        try:
            from app.services.ai import generate_ai_response
            response = await generate_ai_response(
                video_id=video_id,
                question=user_question,
                tab_id=sid
            )
            
            # CRITICAL FIX: Check if this request is still active before responding
            # And send only ONE response with completion flag
            if sid in session_data and 'active_requests' in session_data[sid] and message_id in session_data[sid]['active_requests']:
                await sio.emit('ai_response', {
                    'status': 'success',
                    'answer': response.get('answer', 'I couldn\'t generate a response'),
                    'timestamps': response.get('timestamps', []),
                    'question': user_question,
                    'video_id': video_id,
                    'message_id': message_id,
                    'is_complete': True  # Add completion flag here
                }, to=sid)
                
                # Remove this request from active requests
                session_data[sid]['active_requests'].remove(message_id)
            else:
                logger.info(f"Request {message_id} was cancelled or session expired, not sending response")
        except ImportError as import_error:
            logger.error(f"Failed to import AI module: {str(import_error)}")
            
            # CRITICAL FIX: Check if this request is still active before responding
            # And send only ONE response with completion flag
            if sid in session_data and 'active_requests' in session_data[sid] and message_id in session_data[sid]['active_requests']:
                await sio.emit('ai_response', {
                    'status': 'error',
                    'answer': "I'm sorry, but the AI service is currently unavailable.",
                    'timestamps': [],
                    'question': user_question,
                    'video_id': video_id,
                    'message_id': message_id,
                    'is_complete': True  # Add completion flag here
                }, to=sid)
                
                # Remove this request from active requests
                session_data[sid]['active_requests'].remove(message_id)
            else:
                logger.info(f"Request {message_id} was cancelled or session expired, not sending response")
        
    except Exception as e:
        logger.error(f"Error in legacy ask_ai: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('error', {'message': f'An error occurred: {str(e)}'}, to=sid)
        
        # Send error response with completion flag
        await sio.emit('ai_response', {
            'status': 'error',
            'answer': f'An error occurred while processing your request: {str(e)}',
            'video_id': video_id if 'video_id' in locals() else None,
            'message_id': message_id if 'message_id' in locals() else None,
            'is_complete': True
        }, to=sid)

# Added a new function to generate fallback timestamps from transcript text
async def generate_fallback_timestamps(transcript_text, video_id):
    """
    Generate simple timestamps from transcript text when API methods fail
    This is a fallback mechanism to ensure users always get some timestamps
    
    Args:
        transcript_text: Full transcript text
        video_id: Video ID for reference
        
    Returns:
        dict: Dictionary with formatted_timestamps and raw_timestamps
    """
    if not transcript_text or transcript_text.strip() == "":
        return {"formatted_timestamps": [], "raw_timestamps": []}
        
    logger.info(f"Generating fallback timestamps for video_id: {video_id}")
    
    # Split transcript by sentences or periods for basic segmentation
    # Improved to handle multiple punctuation marks
    segments = re.split(r'[.!?]+\s+', transcript_text.strip())
    segments = [s.strip() for s in segments if s.strip()]
    
    # If the transcript is very short, just use the whole thing
    if len(segments) <= 1:
        segments = [transcript_text.strip()]
    
    # Calculate approximately 13 timestamps (as seen in your logs)
    num_segments = len(segments)
    
    # If we have very few segments, we might need to merge them for reasonable timestamps
    if num_segments < 5:
        logger.info(f"Too few segments ({num_segments}), using whole transcript")
        segments = [transcript_text.strip()]
        num_segments = 1
    
    # Assume a 5-minute video (300 seconds) duration if we can't determine it
    # Can be improved with actual video duration if available
    estimated_duration = 300  
    
    # Try to get a better duration estimate from the database if available
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        with get_db_context() as db:
            result = db.execute(
                text("SELECT duration FROM videos WHERE id::text = :id"),
                {"id": str(video_id)}
            )
            video_row = result.fetchone()
            if video_row and video_row[0]:
                estimated_duration = float(video_row[0])
                logger.info(f"Using actual video duration: {estimated_duration} seconds")
    except Exception as e:
        logger.warning(f"Could not get video duration from database: {str(e)}")
    
    # Generate timestamps with even spacing
    segment_duration = estimated_duration / (num_segments or 1)
    
    # Generate formatted timestamps
    formatted_timestamps = []
    raw_timestamps = []
    
    for i, segment in enumerate(segments):
        # Skip empty segments
        if not segment.strip():
            continue
            
        # Calculate start and end times
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        # Format the timestamp display
        start_minutes = int(start_time // 60)
        start_seconds = int(start_time % 60)
        end_minutes = int(end_time // 60)
        end_seconds = int(end_time % 60)
        
        display_time = f"{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}"
        
        # Add to formatted timestamps list
        formatted_timestamps.append({
            "text": segment[:60] + ("..." if len(segment) > 60 else ""),
            "start_time": start_time,
            "end_time": end_time,
            "display_time": display_time
        })
        
        # Create word-level timestamps for visualization
        words = segment.split()
        word_duration = segment_duration / (len(words) or 1)
        
        for j, word in enumerate(words):
            word_start = start_time + (j * word_duration)
            word_end = start_time + ((j + 1) * word_duration)
            
            raw_timestamps.append({
                "word": word,
                "start_time": word_start,
                "end_time": word_end
            })
    
    logger.info(f"Generated {len(formatted_timestamps)} fallback timestamps")
    
    # Save these timestamps for future use
    try:
        timestamp_dir = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps")
        os.makedirs(timestamp_dir, exist_ok=True)
        timestamp_file = os.path.join(timestamp_dir, f"{video_id}_timestamps.json")
        
        with open(timestamp_file, "w") as f:
            json.dump({
                "formatted_timestamps": formatted_timestamps,
                "raw_timestamps": raw_timestamps
            }, f)
        logger.info(f"Saved fallback timestamps to {timestamp_file}")
    except Exception as e:
        logger.error(f"Error saving fallback timestamps: {str(e)}")
    
    return {
        "formatted_timestamps": formatted_timestamps,
        "raw_timestamps": raw_timestamps
    }

# IMPROVED: Enhanced load_transcription function to handle both file_id and upload_id patterns
async def load_transcription(video_id: str) -> str:
    """Load transcription from saved file with improved error handling"""
    # Standardize video_id format - remove any spaces or special chars
    safe_video_id = re.sub(r'[^\w-]', '_', str(video_id))
    
    # Get standardized IDs
    _, upload_id, file_id = standardize_video_id(video_id)
    
    # Try multiple file naming patterns
    transcription_paths = [
        # Standard patterns with original ID
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json"),
        # Upload ID patterns
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{upload_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{upload_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{upload_id}.json"),
        # File ID patterns 
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{file_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{file_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{file_id}.json"),
        # Sanitized patterns 
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{safe_video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{safe_video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{safe_video_id}.json"),
    ]
    
    # Try each possible file path
    for path in transcription_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Found transcription file at: {path}")
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Safely get text with default
                        return data.get("text", "")
        except Exception as e:
            logger.error(f"Error loading transcription from {path}: {str(e)}")
    
    # If no file found, try wildcard search
    try:
        import glob
        
        # Create multiple wildcard patterns
        wildcard_patterns = [
            os.path.join(settings.TRANSCRIPTION_DIR, f"*{video_id}*.json"),
            os.path.join(settings.TRANSCRIPTION_DIR, f"*{upload_id}*.json"),
            os.path.join(settings.TRANSCRIPTION_DIR, f"*{file_id}*.json")
        ]
        
        for pattern in wildcard_patterns:
            matching_files = glob.glob(pattern)
            
            if matching_files:
                logger.info(f"Found potential transcription match using wildcard: {matching_files[0]}")
                with open(matching_files[0], "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "text" in data:
                        return data.get("text", "")
    except Exception as e:
        logger.error(f"Error in wildcard search: {str(e)}")
    
    # Try database as fallback with improved reliability
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        with get_db_context() as db:
            # Try multiple ID patterns
            id_patterns = [video_id, upload_id, file_id]
            
            for pattern in id_patterns:
                # Try exact match first
                result = db.execute(
                    text("SELECT transcription FROM videos WHERE id::text = :id"),
                    {"id": str(pattern)}
                )
                video_row = result.fetchone()
                if video_row and video_row[0]:
                    logger.info(f"Found transcription in database with exact match for {pattern}")
                    return video_row[0]
                
                # Try with LIKE query for partial matches
                like_result = db.execute(
                    text("SELECT transcription FROM videos WHERE id::text LIKE :pattern"),
                    {"pattern": f"%{pattern}%"}
                )
                like_row = like_result.fetchone()
                
                if like_row and like_row[0]:
                    logger.info(f"Found transcription in database with LIKE pattern for {pattern}")
                    return like_row[0]
            
            # If video_id starts with "youtube_", try extracting just the YouTube ID part
            if video_id.startswith("youtube_"):
                parts = video_id.split("_")
                if len(parts) >= 2:
                    youtube_id_part = parts[1]
                    logger.info(f"Trying database search for YouTube ID part: {youtube_id_part}")
                    yt_result = db.execute(
                        text("SELECT transcription FROM videos WHERE youtube_id = :yt_id"),
                        {"yt_id": youtube_id_part}
                    )
                    yt_row = yt_result.fetchone()
                    
                    if yt_row and yt_row[0]:
                        logger.info(f"Found transcription by YouTube ID")
                        return yt_row[0]
    except Exception as e:
        logger.error(f"Error loading transcription from database: {str(e)}")
    
    return ""

@sio.event
async def fetch_transcription(sid, data):
    """
    Handle requests to fetch video transcription with improved error handling
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received fetch_transcription request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        tab_id = data.get('tabId')
        
        if not video_id:
            await sio.emit('error', {'message': 'Missing video_id'}, to=sid)
            return
        
        # Log the request with video_id
        logger.info(f"Fetching transcription for video_id: {video_id}, tabId: {tab_id}")
        
        # Store tab_id mapping if provided
        if tab_id:
            tab_to_sid_mapping[tab_id] = sid
            logger.info(f"Mapped tab_id {tab_id} to sid {sid}")
            
            # Also store in session data
            if sid not in session_data:
                session_data[sid] = {}
            session_data[sid]['tab_id'] = tab_id
        
        # Clear cache for this video if available
        if hasattr(cache, 'clear_cache') and video_id:
            try:
                cache.clear_cache(video_id)
                logger.info(f"Cleared cache for video_id: {video_id}")
            except Exception as cache_error:
                logger.warning(f"Error clearing cache: {str(cache_error)}")
        
        # Get standardized IDs
        _, upload_id, file_id = standardize_video_id(video_id)
        logger.info(f"Standardized IDs - upload_id: {upload_id}, file_id: {file_id}")
        
        # Try to find an existing transcription with improved search
        transcript = None
        transcription_file = None
        has_timestamps = False
        
        # If this is an upload ID, focus on the hash part which is most reliable
        if video_id and video_id.startswith("upload_"):
            parts = video_id.split("_")
            if len(parts) >= 3:
                hash_part = parts[2]
                logger.info(f"This is an upload ID, extracted hash: {hash_part}")
                
                # First try the exact path
                exact_path = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
                if os.path.exists(exact_path):
                    try:
                        with open(exact_path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                transcript = data.get("text", "")
                                transcription_file = exact_path
                                # Check if timestamps data might exist
                                has_timestamps = "timestamps" in data or "transcript_with_timestamps" in data
                                logger.info(f"Found transcription at exact path: {exact_path}")
                    except Exception as e:
                        logger.error(f"Error reading exact match file: {str(e)}")
                
                # If not found by exact path, look for any files with this hash part
                if not transcript:
                    try:
                        for file in os.listdir(settings.TRANSCRIPTION_DIR):
                            if hash_part in file and file.endswith(".json"):
                                file_path = os.path.join(settings.TRANSCRIPTION_DIR, file)
                                try:
                                    with open(file_path, "r") as f:
                                        data = json.load(f)
                                        if isinstance(data, dict):
                                            transcript = data.get("text", "")
                                            transcription_file = file_path
                                            # Check if timestamps data might exist
                                            has_timestamps = "timestamps" in data or "transcript_with_timestamps" in data
                                            logger.info(f"Found transcription by hash part: {file_path}")
                                            break
                                except Exception as e:
                                    logger.error(f"Error reading potential match file: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error searching for files by hash: {str(e)}")
        
        # If no transcript found by hash matching, try standard paths
        if not transcript:
            # Try with the enhanced load_transcription function
            transcript = await load_transcription(video_id)
            if transcript:
                logger.info(f"Found transcription using enhanced loader, length: {len(transcript)} chars")
                
                # Check for timestamps
                timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
                has_timestamps = os.path.exists(timestamp_path)
                if has_timestamps:
                    logger.info(f"Found timestamps at {timestamp_path}")
        
        # If still not found, check if the video is currently being processed
        if (not transcript or transcript == "") and video_id:
            # Check if we have debug mode enabled
            if debug_mode_enabled:
                logger.info(f"Debug mode is enabled, creating a mock transcription for debugging")
                mock_result = {
                    "text": f"This is a debug transcription for video ID: {video_id} (created on demand)",
                    "segments": [
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 5.0,
                            "text": f"This is a debug transcription for video ID: {video_id} (created on demand)"
                        }
                    ],
                    "timestamps": [
                        {"word": "This", "start_time": 0.0, "end_time": 0.5},
                        {"word": "is", "start_time": 0.5, "end_time": 0.7},
                        {"word": "a", "start_time": 0.7, "end_time": 0.9},
                        {"word": "debug", "start_time": 0.9, "end_time": 1.2},
                        {"word": "transcription", "start_time": 1.2, "end_time": 2.0}
                    ]
                }
                
                # Save this mock result for future requests
                mock_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
                try:
                    os.makedirs(os.path.dirname(mock_file), exist_ok=True)
                    with open(mock_file, "w") as f:
                        json.dump(mock_result, f)
                    logger.info(f"Created and saved mock transcription to {mock_file}")
                    
                    # Use this as the transcript
                    transcript = mock_result["text"]
                    has_timestamps = True
                except Exception as e:
                    logger.error(f"Error saving mock transcription: {e}")
            else:
                # In production mode, check if transcription is still processing
                status_message = f"No transcription found for this video (ID: {video_id}). You may need to process it first."
                
                # Try to fetch any processing info
                from app.services.task_queue import task_queue
                all_tasks = task_queue.tasks.values() if hasattr(task_queue, 'tasks') else []
                
                for task in all_tasks:
                    if task.task_type == "transcribe_video" and task.params.get("video_id") == video_id:
                        if task.status == "pending":
                            status_message = f"Transcription for video (ID: {video_id}) is queued and waiting to be processed."
                        elif task.status == "running":
                            status_message = f"Transcription for video (ID: {video_id}) is currently being processed."
                        break
                
                transcript = status_message
        
        # If still not found, create a fallback message
        if not transcript or transcript == "":
            # IMPORTANT: Include the FULL video_id in the error message
            transcript = f"No transcription found for this video (ID: {video_id}). You may need to process it first."
        
        # Log information about the transcript
        if transcription_file:
            logger.info(f"Using transcription from file: {transcription_file}")
        elif transcript:
            logger.info(f"Using transcription from database or fallback, length: {len(transcript)} chars")
        else:
            logger.warning(f"No transcription found for video_id: {video_id}")
            transcript = "No transcription available for this video."
            
        # Send transcript back to client with video_id - Use BOTH original and standardized IDs
        await sio.emit('transcription', {
            'status': 'success',
            'video_id': video_id,
            'standardized_id': upload_id,
            'file_id': file_id,
            'transcript': transcript,
            'has_timestamps': has_timestamps
        }, to=sid)
        
        # If timestamps are available, notify the client
        if has_timestamps:
            await sio.emit('timestamps_available', {
                'video_id': video_id,
                'standardized_id': upload_id,
                'file_id': file_id
            }, to=sid)
        else:
            # If transcript was found but no timestamps, generate fallback timestamps
            if transcript and transcript != "No transcription available for this video." and not transcript.startswith("No transcription found"):
                # Generate fallback timestamps asynchronously
                asyncio.create_task(async_generate_and_send_timestamps(sid, video_id, transcript))
                
        logger.info(f"Sent transcription for video_id: {video_id}, has_timestamps: {has_timestamps}")
        
    except Exception as e:
        logger.error(f"Error fetching transcription: {str(e)}", exc_info=True)
        await sio.emit('error', {'message': f'Error fetching transcription: {str(e)}'}, to=sid)
        # Also send a simple fallback transcription
        await sio.emit('transcription', {
            'status': 'error',
            'video_id': video_id if 'video_id' in locals() else "unknown",
            'transcript': f"Error loading transcription: {str(e)}",
            'has_timestamps': False
        }, to=sid)

# New helper function to generate and send timestamps asynchronously
async def async_generate_and_send_timestamps(sid, video_id, transcript):
    """Generate timestamps from transcript and send to client asynchronously"""
    try:
        logger.info(f"Generating fallback timestamps for {video_id} in background")
        
        # Generate timestamps
        timestamps_data = await generate_fallback_timestamps(transcript, video_id)
        
        # Get standardized IDs
        _, upload_id, file_id = standardize_video_id(video_id)
        
        # Send notification that timestamps are now available
        await sio.emit('timestamps_available', {
            'video_id': video_id,
            'standardized_id': upload_id,
            'file_id': file_id
        }, to=sid)
        
        # Also send the timestamps directly
        await sio.emit('transcript_timestamps', {
            'status': 'success',
            'video_id': video_id,
            'standardized_id': upload_id,
            'file_id': file_id,
            'timestamps': timestamps_data.get("formatted_timestamps", []),
            'source': 'fallback_generation'
        }, to=sid)
        
        logger.info(f"Sent {len(timestamps_data.get('formatted_timestamps', []))} timestamps")
    except Exception as e:
        logger.error(f"Error generating fallback timestamps: {str(e)}")
        # Don't send an error to the client - this is a background task

# COMPLETELY REWRITTEN YOUTUBE VIDEO PROCESSING FUNCTION
@sio.event
async def process_youtube_video_handler(sid, data):
    """
    Handle request to transcribe a YouTube video using the dedicated service function
    
    This is a completely rewritten version that prioritizes user experience
    by providing a transcription even if the download fails.
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received YouTube transcription request from client {client_id}: {data}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            logger.error(f"Invalid data format: {data}")
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        tab_id = data.get('tabId')
        youtube_url = data.get('youtube_url')
        
        # Store tab_id mapping if provided
        if tab_id:
            tab_to_sid_mapping[tab_id] = sid
            logger.info(f"Mapped tab_id {tab_id} to sid {sid}")
            
            # Also store in session data
            if sid not in session_data:
                session_data[sid] = {}
            session_data[sid]['tab_id'] = tab_id
        
        # Extract YouTube URL from data if not directly provided
        if not youtube_url and data.get('url'):
            youtube_url = data.get('url')
            logger.info(f"Using 'url' field instead of 'youtube_url': {youtube_url}")
        
        # Check for debug mode
        debug_mode = data.get('debug_mode', False)
        
        # Only use debug mode if explicitly requested or set in environment
        if not debug_mode:
            debug_mode = debug_mode_enabled
        
        # Validate the YouTube URL
        if not youtube_url or not isinstance(youtube_url, str):
            logger.error(f"Missing or invalid YouTube URL: {youtube_url}")
            await sio.emit('error', {'message': 'Please enter a valid YouTube URL'}, to=sid)
            return
            
        # Clean and validate the URL
        youtube_url = youtube_url.strip()
        if not youtube_url.startswith(('https://www.youtube.com/', 'https://youtu.be/', 'http://www.youtube.com/', 'http://youtu.be/')):
            logger.error(f"Invalid YouTube URL format: {youtube_url}")
            await sio.emit('error', {'message': 'Please enter a valid YouTube URL'}, to=sid)
            return
            
        # Get or generate video_id
        video_id = data.get('video_id')
        if not video_id:
            # Generate a unique video ID if not provided
            timestamp = int(time.time())
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
            video_id = f"youtube_{timestamp}_{random_id}"
            logger.info(f"Generated new video_id: {video_id}")
        
        # Clear any existing cache for this video_id if it exists
        if hasattr(cache, 'clear_cache') and video_id:
            try:
                cache.clear_cache(video_id)
                logger.info(f"Cleared cache for video_id: {video_id}")
            except Exception as cache_error:
                logger.warning(f"Error clearing cache: {str(cache_error)}")
        
        # Detect YouTube video type and get settings
        video_type, timeout_duration, priority = detect_youtube_video_type(youtube_url)
        logger.info(f"Detected YouTube video type: {video_type}, timeout: {timeout_duration}s, priority: {priority}")
        
        # Store the video ID in session data
        if sid not in session_data:
            session_data[sid] = {}
        session_data[sid]['current_video_id'] = video_id
        
        # Acknowledge receipt of the request immediately
        await sio.emit('transcription_status', {
            'status': 'received',
            'message': f'Request received and processing started for {video_type} video...',
            'video_id': video_id,
            'video_type': video_type
        }, to=sid)
        
        # Ensure output directory exists
        os.makedirs(settings.TRANSCRIPTION_DIR, exist_ok=True)
        output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
        temp_video_path = os.path.join(settings.TEMP_DIR, f"{video_id}_video.mp4")
        
        # Set up progress updates
        progress_task = asyncio.create_task(send_periodic_progress_updates(sid, video_id))
        
        try:
            # Try to download the video
            await sio.emit('transcription_status', {
                'status': 'downloading',
                'message': 'Downloading and processing video...',
                'video_id': video_id,
                'progress': 10
            }, to=sid)
            
            # Use our improved download function with enhanced reliability
            download_success, download_message, metadata = await download_youtube_video(
                youtube_url=youtube_url,
                output_path=temp_video_path,
                sid=sid
            )
            
            # If download failed, but we got metadata, we'll create a synthetic transcription
            if not download_success and metadata:
                logger.warning(f"YouTube download failed, creating synthetic transcription: {download_message}")
                
                # Notify user about the issue
                await sio.emit('transcription_status', {
                    'status': 'warning',
                    'message': f'Note: {download_message}. Creating transcription with available info.',
                    'video_id': video_id,
                    'progress': 50
                }, to=sid)
                
                # Use our function to create a synthetic transcription
                result = create_synthetic_transcription(video_id, youtube_url, metadata)
                
                # Save the result for future use
                try:
                    with open(output_file, "w") as f:
                        json.dump(result, f)
                    logger.info(f"Saved synthetic transcription to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving synthetic transcription: {e}")
            
            # If download succeeded or we're in debug mode, try the transcription service
            elif download_success or debug_mode:
                await sio.emit('transcription_status', {
                    'status': 'transcribing',
                    'message': 'Processing YouTube video...',
                    'video_id': video_id,
                    'progress': 30
                }, to=sid)
                
                try:
                    # Try the transcription service with timeout
                    result = await asyncio.wait_for(
                        transcribe_youtube_video_service(
                            youtube_url=youtube_url,
                            output_file=output_file,
                            video_id=video_id,
                            language_code="en",
                            debug_mode=debug_mode
                        ),
                        timeout=timeout_duration
                    )
                    
                    # If result is invalid, create synthetic result
                    if not result or not isinstance(result, dict) or "text" not in result:
                        logger.warning(f"Invalid transcription result, creating synthetic version")
                        result = create_synthetic_transcription(video_id, youtube_url, metadata)
                        
                        # Save the synthetic result
                        with open(output_file, "w") as f:
                            json.dump(result, f)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Transcription service timed out, using synthetic transcription")
                    result = create_synthetic_transcription(video_id, youtube_url, metadata)
                    
                    # Save the synthetic result
                    with open(output_file, "w") as f:
                        json.dump(result, f)
                    
                except Exception as service_error:
                    logger.error(f"Transcription service error: {str(service_error)}")
                    result = create_synthetic_transcription(video_id, youtube_url, metadata)
                    
                    # Save the synthetic result
                    with open(output_file, "w") as f:
                        json.dump(result, f)
            else:
                # Both download and service failed, create synthetic result
                logger.warning(f"Complete transcription failure, using synthetic fallback")
                result = create_synthetic_transcription(video_id, youtube_url, {})
                
                # Save the synthetic result
                with open(output_file, "w") as f:
                    json.dump(result, f)
            
            # Cancel progress updates
            if not progress_task.done():
                progress_task.cancel()
            
            # Make sure we have timestamps (either from result or generated)
            formatted_timestamps = result.get("formatted_timestamps", [])
            raw_timestamps = result.get("timestamps", [])
            
            if not formatted_timestamps and not raw_timestamps:
                # Generate timestamps from text
                timestamps_data = await generate_fallback_timestamps(result["text"], video_id)
                formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
                raw_timestamps = timestamps_data.get("raw_timestamps", [])
                
                # Update the result
                result["formatted_timestamps"] = formatted_timestamps
                result["timestamps"] = raw_timestamps
                
                # Save the updated result
                with open(output_file, "w") as f:
                    json.dump(result, f)
            
            # Send completion status
            await sio.emit('transcription_status', {
                'status': 'completed',
                'message': 'Transcription process completed',
                'video_id': video_id,
                'progress': 100
            }, to=sid)
            
            # Send the transcript to the client
            await sio.emit('transcription', {
                'status': 'success',
                'video_id': video_id,
                'transcript': result["text"],
                'has_timestamps': True
            }, to=sid)
            
            # Always send timestamps available notification
            await sio.emit('timestamps_available', {
                'video_id': video_id
            }, to=sid)
            
            # Also send the timestamps directly
            await sio.emit('transcript_timestamps', {
                'status': 'success',
                'video_id': video_id,
                'timestamps': formatted_timestamps or raw_timestamps
            }, to=sid)
            
            logger.info(f"Completed YouTube video processing for {video_id}")
            
        except Exception as e:
            # Cancel progress updates if still running
            if 'progress_task' in locals() and not progress_task.done():
                progress_task.cancel()
            
            # Handle any unexpected errors
            logger.error(f"Unexpected error in YouTube processing: {str(e)}", exc_info=True)
            
            # Create a basic error response
            error_text = f"Error processing YouTube video: {str(e)}"
            error_result = create_synthetic_transcription(
                video_id=video_id,
                url=youtube_url,
                metadata={"error": str(e)}
            )
            
            # Save the error result
            try:
                with open(output_file, "w") as f:
                    json.dump(error_result, f)
            except Exception as write_error:
                logger.error(f"Error saving error transcription: {write_error}")
            
            # Send error status but with transcription data
            await sio.emit('transcription_status', {
                'status': 'completed',
                'message': 'Transcription process completed with errors',
                'video_id': video_id,
                'progress': 100
            }, to=sid)
            
            # Send the transcript with error info but don't show raw error to user
            await sio.emit('transcription', {
                'status': 'success',  # Still mark as success to avoid UI problems
                'video_id': video_id,
                'transcript': error_result["text"],
                'has_timestamps': True
            }, to=sid)
            
            # Send timestamps available notification
            await sio.emit('timestamps_available', {
                'video_id': video_id
            }, to=sid)
            
            # Send the timestamps
            await sio.emit('transcript_timestamps', {
                'status': 'success',
                'video_id': video_id,
                'timestamps': error_result["formatted_timestamps"]
            }, to=sid)
    
    except Exception as outer_error:
        # This is a catastrophic error - handle it gracefully
        logger.error(f"Catastrophic error in process_youtube_video_handler: {str(outer_error)}", exc_info=True)
        
        # Try to send a minimal error notification
        try:
            await sio.emit('error', {
                'message': 'An unexpected error occurred. Please try again.'
            }, to=sid)
            
            # If we have a video_id, send error statuses
            if 'video_id' in locals():
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': 'Transcription failed. Please try again.',
                    'video_id': video_id,
                    'progress': 100
                }, to=sid)
                
                # Create a minimal error transcription
                error_text = "The transcription process failed. Please try again with a different video or URL."
                await sio.emit('transcription', {
                    'status': 'error',
                    'video_id': video_id,
                    'transcript': error_text,
                    'has_timestamps': False
                }, to=sid)
        except Exception as emit_error:
            logger.error(f"Failed to send error notification: {str(emit_error)}")

# Create alias functions for different event names that all point to the same handler
@sio.event
async def transcribe_youtube_process(sid, data):
    """Handle direct YouTube transcription requests - wrapper for process_youtube_video_handler"""
    logger.info(f"Received transcribe_youtube_process request, forwarding to process_youtube_video_handler")
    await process_youtube_video_handler(sid, data)

@sio.event
async def process_youtube(sid, data):
    """Handle process_youtube requests - wrapper for process_youtube_video_handler"""
    logger.info(f"Received process_youtube request, forwarding to process_youtube_video_handler")
    await process_youtube_video_handler(sid, data)

@sio.event
async def analyze_youtube(sid, data):
    """Handle analyze_youtube requests - wrapper for process_youtube_video_handler"""
    logger.info(f"Received analyze_youtube request, forwarding to process_youtube_video_handler")
    await process_youtube_video_handler(sid, data)

@sio.event
async def transcribe_youtube_video(sid, data):
    """Handle transcribe_youtube_video requests - wrapper for process_youtube_video_handler"""
    logger.info(f"Received transcribe_youtube_video request, forwarding to process_youtube_video_handler")
    await process_youtube_video_handler(sid, data)

# New helper function to send periodic progress updates
async def send_periodic_progress_updates(sid, video_id):
    """Send periodic progress updates to keep the connection alive during long operations"""
    try:
        progress = 10
        max_progress = 45  # Cap progress at 45% to leave room for subsequent steps
        
        while progress < max_progress:
            # Wait between updates
            await asyncio.sleep(3)
            
            # Increment progress
            progress += 5
            if progress > max_progress:
                progress = max_progress
                
            # Send update
            message = 'Downloading and processing video...'
            if progress > 30:
                message = 'Extracting audio for transcription...'
                
            try:
                await sio.emit('transcription_status', {
                    'status': 'processing',
                    'message': message,
                    'video_id': video_id,
                    'progress': progress
                }, to=sid)
                
                # Also send a heartbeat to keep the connection alive
                await sio.emit('heartbeat', {'timestamp': time.time()}, to=sid)
                
            except Exception as emit_error:
                logger.error(f"Error sending progress update: {str(emit_error)}")
                
    except asyncio.CancelledError:
        logger.info(f"Progress updates cancelled for {video_id}")
        
    except Exception as e:
        logger.error(f"Error in progress updates: {str(e)}")

# New function to send continuous heartbeat to keep connections alive
async def send_heartbeat(sid):
    """Send regular heartbeat to prevent timeouts"""
    try:
        count = 0
        while sid in session_data:
            await asyncio.sleep(25)  # Send heartbeat every 25 seconds
            
            try:
                if sid in session_data:
                    count += 1
                    await sio.emit('heartbeat', {
                        'timestamp': time.time(),
                        'count': count
                    }, to=sid)
            except Exception as e:
                logger.error(f"Error sending heartbeat to {sid}: {str(e)}")
                break
                
    except asyncio.CancelledError:
        logger.debug(f"Heartbeat task cancelled for {sid}")
    except Exception as e:
        logger.error(f"Error in heartbeat task for {sid}: {str(e)}")

# IMPROVED: Process Uploaded Video Handler with proper ID handling
@sio.event
async def process_uploaded_video(sid, data):
    """
    Process transcription for an uploaded video file
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received uploaded video processing request from client {client_id}")
    
    try:
        # Validate input data with better logging
        if not isinstance(data, dict):
            logger.error(f"Invalid data format: {data}")
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        video_path = data.get('video_path')
        tab_id = data.get('tabId')
        
        # Store tab_id mapping if provided
        if tab_id:
            tab_to_sid_mapping[tab_id] = sid
            logger.info(f"Mapped tab_id {tab_id} to sid {sid}")
            
            # Also store in session data
            if sid not in session_data:
                session_data[sid] = {}
            session_data[sid]['tab_id'] = tab_id
        
        # Validate required parameters
        if not video_id:
            logger.error("Missing video_id parameter")
            await sio.emit('error', {'message': 'Missing video_id parameter'}, to=sid)
            return
            
        if not video_path:
            logger.error("Missing video_path parameter")
            await sio.emit('error', {'message': 'Missing video_path parameter'}, to=sid)
            return
            
        # Check if the video file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            
            # Try to find the file with a case-insensitive search
            dir_path = os.path.dirname(video_path)
            filename = os.path.basename(video_path)
            
            found = False
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.lower() == filename.lower():
                        video_path = os.path.join(dir_path, file)
                        logger.info(f"Found file with case-insensitive match: {video_path}")
                        found = True
                        break
            
            if not found:
                await sio.emit('error', {'message': f'Video file not found: {video_path}'}, to=sid)
                return
            
        # Clear cache for this video if available
        if hasattr(cache, 'clear_cache') and video_id:
            try:
                cache.clear_cache(video_id)
                logger.info(f"Cleared cache for video_id: {video_id}")
            except Exception as cache_error:
                logger.warning(f"Error clearing cache: {str(cache_error)}")
                
        # Get standardized IDs
        standardized_id, upload_id, file_id = standardize_video_id(video_id)
        logger.info(f"Standardized IDs - upload_id: {upload_id}, file_id: {file_id}")
        
        # Check for debug mode explicitly
        debug_mode = data.get('debug_mode', False)
        # Only use debug mode if explicitly requested or set in environment
        if not debug_mode:
            debug_mode = debug_mode_enabled
            if debug_mode:
                logger.info("Using debug mode from environment variable")
            else:
                logger.info("Using real transcription service (debug mode off)")
            
        if debug_mode:
            logger.info(f"🐞 DEBUG MODE ENABLED for uploaded video transcription: {video_id}")
        else:
            logger.info(f"🚀 Using REAL transcription for uploaded video: {video_id}")
        
        # Log basic information
        logger.info(f"Processing uploaded video: ID={video_id}, Path={video_path}, TabID={tab_id}")
        
        # Store video ID in session data
        if sid not in session_data:
            session_data[sid] = {}
        session_data[sid]['current_video_id'] = video_id
        
        # Acknowledge receipt of the request immediately
        await sio.emit('transcription_status', {
            'status': 'received',
            'message': 'Request received and processing started for uploaded video...',
            'video_id': video_id,
            'upload_id': upload_id,
            'file_id': file_id
        }, to=sid)
        
        # Ensure output directory exists
        os.makedirs(settings.TRANSCRIPTION_DIR, exist_ok=True)
        output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{upload_id}.json")
        
        # Start progress updates
        progress_task = asyncio.create_task(send_periodic_progress_updates(sid, upload_id))
        
        try:
            # Update status to transcribing
            await sio.emit('transcription_status', {
                'status': 'transcribing',
                'message': 'Transcribing uploaded video...',
                'video_id': video_id,
                'upload_id': upload_id,
                'file_id': file_id,
                'progress': 30
            }, to=sid)
            
            # Use the transcribe_video service function with timeout
            result = await asyncio.wait_for(
                transcribe_video_service(  # Use imported function with different name
                    video_path=video_path,
                    output_file=output_file,
                    video_id=upload_id,
                    debug_mode=debug_mode  # Pass the debug flag
                ),
                timeout=1200  # 20-minute timeout for uploaded videos
            )
            
            # Cancel the progress updates
            progress_task.cancel()
            
            # Check if transcription was successful
            if result and isinstance(result, dict) and "text" in result:
                logger.info(f"Uploaded video transcription completed successfully for video_id: {upload_id}")
                
                # Generate timestamps if result doesn't have them
                formatted_timestamps = []
                raw_timestamps = []
                
                # Check if result has timestamps already
                if "timestamps" not in result or not result["timestamps"]:
                    try:
                        # Generate timestamps from text
                        timestamps_data = await generate_fallback_timestamps(result["text"], upload_id)
                        formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
                        raw_timestamps = timestamps_data.get("raw_timestamps", [])
                        
                        # Add timestamps to result
                        result["timestamps"] = raw_timestamps
                        result["formatted_timestamps"] = formatted_timestamps
                        
                        # Save the updated result
                        with open(output_file, "w") as f:
                            json.dump(result, f)
                        
                        logger.info(f"Added generated timestamps to transcription result")
                    except Exception as ts_error:
                        logger.error(f"Error generating timestamps: {str(ts_error)}")
                else:
                    # Use existing timestamps
                    if "formatted_timestamps" in result:
                        formatted_timestamps = result["formatted_timestamps"]
                    else:
                        # Generate formatted timestamps from raw ones if needed
                        # This ensures we have both formats
                        raw_timestamps = result["timestamps"]
                        # We'll use them as they are
                
                # Send completion status
                await sio.emit('transcription_status', {
                    'status': 'completed',
                    'message': 'Transcription completed successfully',
                    'video_id': video_id,
                    'upload_id': upload_id,
                    'file_id': file_id,
                    'progress': 100
                }, to=sid)
                
                # Send the transcript directly
                has_timestamps = bool(formatted_timestamps or raw_timestamps)
                await sio.emit('transcription', {
                    'status': 'success',
                    'video_id': video_id,
                    'upload_id': upload_id,
                    'file_id': file_id,
                    'transcript': result["text"],
                    'has_timestamps': has_timestamps
                }, to=sid)
                
                # Also send to the tab_id if it's available and different from sid
                if tab_id and tab_id in tab_to_sid_mapping and tab_to_sid_mapping[tab_id] != sid:
                    logger.info(f"Also sending transcription to tab ID: {tab_id}")
                    
                    await emit_to_tab('transcription', {
                        'status': 'success',
                        'video_id': video_id,
                        'upload_id': upload_id, 
                        'file_id': file_id,
                        'transcript': result["text"],
                        'has_timestamps': has_timestamps
                    }, tab_id)
                
                # If timestamps are available, notify client
                if has_timestamps:
                    await sio.emit('timestamps_available', {
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id
                    }, to=sid)
                    
                    # IMPORTANT: Pre-send the timestamps data to prevent 404 errors later
                    await sio.emit('transcript_timestamps', {
                        'status': 'success',
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id,
                        'timestamps': formatted_timestamps or raw_timestamps
                    }, to=sid)
                    
                    # Also send to tab_id if needed
                    if tab_id and tab_id in tab_to_sid_mapping and tab_to_sid_mapping[tab_id] != sid:
                        await emit_to_tab('timestamps_available', {
                            'video_id': video_id,
                            'upload_id': upload_id,
                            'file_id': file_id
                        }, tab_id)
                        
                        await emit_to_tab('transcript_timestamps', {
                            'status': 'success',
                            'video_id': video_id,
                            'upload_id': upload_id,
                            'file_id': file_id,
                            'timestamps': formatted_timestamps or raw_timestamps
                        }, tab_id)
            else:
                # Use debug transcription if the real one failed
                if debug_mode:
                    logger.info(f"Using debug transcription for uploaded video: {upload_id}")
                    
                    # Create a mock transcription
                    mock_result = {
                        "text": f"This is a debug transcription for uploaded video ID: {upload_id}",
                        "segments": [
                            {
                                "id": 0,
                                "start": 0.0,
                                "end": 5.0,
                                "text": f"This is a debug transcription for uploaded video ID: {upload_id}"
                            }
                        ],
                        "timestamps": [
                            {"word": "This", "start_time": 0.0, "end_time": 0.5},
                            {"word": "is", "start_time": 0.5, "end_time": 0.7},
                            {"word": "a", "start_time": 0.7, "end_time": 0.9},
                            {"word": "debug", "start_time": 0.9, "end_time": 1.2},
                            {"word": "transcription", "start_time": 1.2, "end_time": 2.0}
                        ]
                    }
                    
                    # Save mock result
                    try:
                        with open(output_file, "w") as f:
                            json.dump(mock_result, f)
                    except Exception as write_error:
                        logger.error(f"Error writing debug transcription: {str(write_error)}")
                    
                    # Send completion status
                    await sio.emit('transcription_status', {
                        'status': 'completed',
                        'message': 'Debug transcription created',
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id,
                        'progress': 100
                    }, to=sid)
                    
                    # Send the mock transcript
                    await sio.emit('transcription', {
                        'status': 'success',
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id,
                        'transcript': mock_result["text"],
                        'has_timestamps': True
                    }, to=sid)
                else:
                    # Handle real transcription failure
                    logger.error(f"Transcription failed for uploaded video: {upload_id}")
                    
                    # Send failure status
                    await sio.emit('transcription_status', {
                        'status': 'error',
                        'message': 'Transcription failed. Please try again or use a different file.',
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id,
                        'progress': 100
                    }, to=sid)
        except asyncio.TimeoutError:
            # Cancel progress updates if still running
            if 'progress_task' in locals() and not progress_task.done():
                progress_task.cancel()
                
            logger.error(f"Transcription timed out for video_id: {upload_id}")
            
            # Create timeout message
            timeout_text = f"Transcription timed out after 20 minutes. The video may be too long or complex."
            
            # Send timeout status
            await sio.emit('transcription_status', {
                'status': 'error',
                'message': 'Transcription timed out. Please try a shorter video.',
                'video_id': video_id,
                'upload_id': upload_id,
                'file_id': file_id,
                'progress': 100
            }, to=sid)
            
            # Send basic transcript with error message
            await sio.emit('transcription', {
                'status': 'error',
                'video_id': video_id,
                'upload_id': upload_id,
                'file_id': file_id,
                'transcript': timeout_text,
                'has_timestamps': False
            }, to=sid)
        except Exception as e:
            # Cancel progress updates if still running
            if 'progress_task' in locals() and not progress_task.done():
                progress_task.cancel()
                
            logger.error(f"Error transcribing uploaded video: {str(e)}", exc_info=True)
            
            # Send error status
            await sio.emit('transcription_status', {
                'status': 'error',
                'message': f'Error during transcription: {str(e)}',
                'video_id': video_id,
                'upload_id': upload_id,
                'file_id': file_id,
                'progress': 100
            }, to=sid)
            
            # Send transcript with error message
            await sio.emit('transcription', {
                'status': 'error',
                'video_id': video_id,
                'upload_id': upload_id,
                'file_id': file_id,
                'transcript': f"Error during transcription: {str(e)}",
                'has_timestamps': False
            }, to=sid)
    except Exception as outer_error:
        # Handle outer exceptions
        logger.error(f"Outer error in process_uploaded_video: {str(outer_error)}", exc_info=True)
        
        # Try to send an error notification
        try:
            await sio.emit('error', {
                'message': 'An unexpected error occurred processing your video.'
            }, to=sid)
            
            # If we have a video_id, send error status
            if 'video_id' in locals():
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': 'Transcription failed due to an unexpected error.',
                    'video_id': video_id,
                    'progress': 100
                }, to=sid)
                
                # Send basic error transcript
                await sio.emit('transcription', {
                    'status': 'error',
                    'video_id': video_id,
                    'transcript': "An unexpected error occurred during video processing. Please try again.",
                    'has_timestamps': False
                }, to=sid)
        except Exception as emit_error:
            logger.error(f"Failed to send error notification: {str(emit_error)}")      
    
# Create additional event handler for the upload_video_processed event
@sio.event
async def upload_video_processed(sid, data):
    """Handle notification that an uploaded video has been processed"""
    logger.info(f"Received upload_video_processed notification, forwarding to process_uploaded_video")
    await process_uploaded_video(sid, data)

@sio.event
async def process_uploaded_audio(sid, data):
    """Process transcription for an uploaded audio file - same as video but with audio flag"""
    # Just call process_uploaded_video with audio flag set to true
    if isinstance(data, dict):
        data['is_audio'] = True
    await process_uploaded_video(sid, data)

@sio.event
async def register_tab(sid, data):
    """
    Handle tab registration and reset any existing state for this tab
    """
    try:
        if not isinstance(data, dict):
            tab_id = data  # Fallback: data could be just the tab ID string
        else:
            tab_id = data.get('tabId') or data.get('tab_id')
        
        if not tab_id:
            return
            
        # Initialize session data if needed
        if sid not in session_data:
            session_data[sid] = {}
            
        # Store client info
        client_id = session_data.get(sid, {}).get('client_id', sid)    
        
        # Store tab_id in session data
        session_data[sid]['tab_id'] = tab_id
        
        # FIXED: Add to mapping to keep track of which sid owns this tab
        tab_to_sid_mapping[tab_id] = sid
        
        logger.info(f"Registered tab ID {tab_id} for SID {sid} (client {client_id})")
        
        # Clear any existing state for this tab/sid to prevent conflicts
        if 'current_video_id' in session_data[sid]:
            old_video_id = session_data[sid]['current_video_id']
            logger.info(f"Clearing previous video state for {old_video_id}")
            session_data[sid]['current_video_id'] = None
        
        if 'current_task_id' in session_data[sid]:
            session_data[sid]['current_task_id'] = None
        
        # Send welcome message to confirm registration
        await sio.emit('welcome', {'message': f'Welcome! Your tab is registered as: {tab_id}'}, to=sid)
        
    except Exception as e:
        logger.error(f"Error in register_tab: {str(e)}", exc_info=True)

@sio.event
async def register_client(sid, data):
    """Register a client ID with this session"""
    try:
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        tab_id = data.get('tab_id')
        client_id = data.get('client_id')
        
        if not client_id:
            return
            
        # Initialize session data if needed
        if sid not in session_data:
            session_data[sid] = {}
            
        # Store client ID
        session_data[sid]['client_id'] = client_id
        
        # Store tab_id if provided
        if tab_id:
            session_data[sid]['tab_id'] = tab_id
            
            # FIXED: Add to mapping
            tab_to_sid_mapping[tab_id] = sid
            
        logger.info(f"Registered client ID {client_id} for SID {sid}")
        
    except Exception as e:
        logger.error(f"Error in register_client: {str(e)}")

# API endpoint to clear transcription cache
@sio.event
async def clear_transcription_cache(sid, data=None):
    """
    Clear the transcription cache
    """
    try:
        if hasattr(cache, 'clear_all_cache'):
            result = cache.clear_all_cache()
            await sio.emit('cache_cleared', {
                'status': 'success' if result else 'error',
                'message': 'Transcription cache cleared successfully' if result else 'Failed to clear cache'
            }, to=sid)
        else:
            # Fallback if clear_all_cache method doesn't exist
            await sio.emit('cache_cleared', {
                'status': 'error',
                'message': 'Cache clearing not implemented'
            }, to=sid)
    except Exception as e:
        logger.error(f"Error clearing transcription cache: {str(e)}")
        await sio.emit('cache_cleared', {
            'status': 'error',
            'message': f'Error clearing cache: {str(e)}'
        }, to=sid)

# Retry YouTube processing with different options
@sio.event
async def retry_youtube_processing(sid, data):
    """
    Retry processing a YouTube video with different options
    
    Args:
        sid: Socket ID
        data: Dictionary with video_id, youtube_url, and options
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received retry request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        youtube_url = data.get('youtube_url')
        tab_id = data.get('tabId')
        
        # Determine if using timestamp
        use_timestamp = data.get('use_timestamp', True)
        
        if not video_id or not youtube_url:
            await sio.emit('error', {'message': 'Missing video_id or youtube_url'}, to=sid)
            return
        
        # Modify URL if using timestamp and not already present
        if use_timestamp and 't=' not in youtube_url and 'start=' not in youtube_url:
            if '?' in youtube_url:
                modified_url = f"{youtube_url}&t=0s"
            else:
                modified_url = f"{youtube_url}?t=0s"
            logger.info(f"Modified URL with timestamp: {modified_url}")
        else:
            modified_url = youtube_url
        
        # Update client on retry attempt
        await sio.emit('transcription_status', {
            'status': 'retrying',
            'message': 'Retrying video processing with modified settings...',
            'video_id': video_id
        }, to=sid)
        
        # Create new data for transcription request
        retry_data = {
            'youtube_url': modified_url,
            'video_id': f"{video_id}_retry_{int(time.time())}",
            'tabId': tab_id,
            'force_download': True,  # Add force download flag
            'bypass_cache': True,    # Add bypass cache flag
        }
        
        # Call the transcribe function with the modified data
        await process_youtube_video_handler(sid, data)
        
    except Exception as e:
        error_msg = f"Error in retry_youtube_processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await sio.emit('error', {'message': error_msg}, to=sid)

# ADDED: Check transcription status handler for the specific 404 error in logs
@sio.event
async def check_transcription_status(sid, data):
    """
    Handle transcription status check requests
    This specifically handles the request pattern causing 404 errors
    """
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        tab_id = data.get('tab_id') or data.get('tabId')
        
        if not video_id:
            logger.error("Missing video_id in check_transcription_status")
            await sio.emit('error', {'message': 'Missing video_id parameter'}, to=sid)
            return
            
        logger.info(f"Received transcription status check for video_id: {video_id}, tab_id: {tab_id}")
        
        # Get standardized IDs
        standard_id, upload_id, file_id = standardize_video_id(video_id)
        logger.info(f"Standardized IDs - standard: {standard_id}, upload: {upload_id}, file: {file_id}")
        
        # Try to find transcription for any of the ID variants
        transcription_files = []
        for id_variant in [standard_id, upload_id, file_id]:
            if not id_variant:
                continue
                
            path = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{id_variant}.json")
            if os.path.exists(path):
                transcription_files.append((id_variant, path))
        
        if transcription_files:
            # Use the first found transcription
            found_id, path = transcription_files[0]
            logger.info(f"Found transcription file for {found_id} at {path}")
            
            # Load the transcription
            try:
                with open(path, 'r') as f:
                    transcription_data = json.load(f)
                    
                # Send the transcription data
                await sio.emit('transcription', {
                    'status': 'success',
                    'video_id': video_id,  # Keep original ID for client reference
                    'upload_id': upload_id,
                    'file_id': file_id,
                    'transcript': transcription_data.get('text', ''),
                    'has_timestamps': 'timestamps' in transcription_data or 'formatted_timestamps' in transcription_data
                }, to=sid)
                
                # Send timestamps if available
                if 'timestamps' in transcription_data or 'formatted_timestamps' in transcription_data:
                    await sio.emit('timestamps_available', {
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id
                    }, to=sid)
                    
                    # Also send the actual timestamps
                    timestamps = transcription_data.get('formatted_timestamps', transcription_data.get('timestamps', []))
                    await sio.emit('transcript_timestamps', {
                        'status': 'success',
                        'video_id': video_id,
                        'upload_id': upload_id,
                        'file_id': file_id,
                        'timestamps': timestamps
                    }, to=sid)
                
                return
            except Exception as e:
                logger.error(f"Error loading transcription from {path}: {e}")
        
        # If no transcription found, check if it's processing
        logger.info("No transcription file found, checking processing status")
        
        # Check task queue for this video
        try:
            from app.services.task_queue import task_queue
            all_tasks = task_queue.tasks.values() if hasattr(task_queue, 'tasks') else []
            
            for task in all_tasks:
                if task.task_type == "transcribe_video":
                    task_video_id = task.params.get("video_id")
                    if task_video_id in [video_id, upload_id, file_id]:
                        if task.status == "pending":
                            status = "Transcription is queued and waiting to be processed."
                        elif task.status == "running":
                            status = "Transcription is currently being processed."
                        else:
                            status = f"Transcription task status: {task.status}"
                            
                        await sio.emit('transcription_status', {
                            'status': 'processing',
                            'message': status,
                            'video_id': video_id,
                            'upload_id': upload_id,
                            'file_id': file_id,
                            'progress': 30
                        }, to=sid)
                        
                        return
        except (ImportError, Exception) as e:
            logger.error(f"Error checking task queue: {e}")
        
        # If we get here, no transcription exists and no task is processing
        # Try to load any cached transcription if available
        try:
            # Check if the video ID might be in a different format
            for id_variant in [standard_id, upload_id, file_id]:
                if not id_variant:
                    continue
                    
                if hasattr(cache, 'get_cached_transcription'):
                    cached_transcription = cache.get_cached_transcription(id_variant)
                    if cached_transcription:
                        logger.info(f"Found cached transcription for {id_variant}")
                        
                        await sio.emit('transcription', {
                            'status': 'success',
                            'video_id': video_id,
                            'upload_id': upload_id,
                            'file_id': file_id,
                            'transcript': cached_transcription.get('text', ''),
                            'has_timestamps': False,
                            'cached': True
                        }, to=sid)
                        
                        return
        except Exception as cache_error:
            logger.error(f"Error checking cache: {cache_error}")
        
        # If still nothing, respond with not found
        await sio.emit('transcription_status', {
            'status': 'not_found',
            'message': f'No transcription found for video ID: {video_id}',
            'video_id': video_id,
            'upload_id': upload_id,
            'file_id': file_id
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error in check_transcription_status: {e}", exc_info=True)
        await sio.emit('error', {'message': f'Error checking transcription status: {str(e)}'}, to=sid)

# NEW HANDLER FOR VISUAL ANALYSIS
@sio.event
async def analyze_visual_youtube(sid, data):
    """
    Handle direct visual analysis requests for YouTube videos
    This specifically handles the format that's causing 404 errors
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received analyze_visual_youtube request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        if not video_id:
            await sio.emit('error', {'message': 'Missing video_id parameter'}, to=sid)
            return
            
        # Log the request
        logger.info(f"Processing visual analysis for YouTube video: {video_id}")
        
        # Let the client know we're processing
        await sio.emit('visual_analysis_status', {
            'status': 'processing',
            'message': 'Processing visual analysis...',
            'video_id': video_id,
            'progress': 10
        }, to=sid)
        
        # Access the database
        try:
            from app.utils.database import get_db_context
            with get_db_context() as db:
                # Try to get visual data for the video
                visual_data = get_visual_data(db, video_id)
                
                # If no data, try to analyze it
                if not visual_data or visual_data.get('status') == 'pending':
                    logger.info(f"No visual data found for {video_id}, generating...")
                    visual_data = analyze_visual_data(db, video_id)
                
                # Update progress
                await sio.emit('visual_analysis_status', {
                    'status': 'processing',
                    'message': 'Processing visual analysis...',
                    'video_id': video_id,
                    'progress': 50
                }, to=sid)
                
                # If still no data, create placeholder
                if not visual_data or not isinstance(visual_data, dict):
                    visual_data = {
                        'status': 'success',
                        'video_id': video_id,
                        'data': {
                            'frames': [],
                            'scenes': [],
                            'topics': [],
                            'highlights': []
                        }
                    }
                
                # Send the completed status
                await sio.emit('visual_analysis_status', {
                    'status': 'completed',
                    'message': 'Visual analysis complete',
                    'video_id': video_id,
                    'progress': 100
                }, to=sid)
                
                # Send the visual data
                await sio.emit('visual_analysis_data', {
                    'status': 'success',
                    'video_id': video_id,
                    'data': visual_data.get('data', {})
                }, to=sid)
                
                logger.info(f"Sent visual analysis data for {video_id}")
                
        except Exception as db_error:
            logger.error(f"Database error in analyze_visual_youtube: {str(db_error)}")
            # Send error status
            await sio.emit('visual_analysis_status', {
                'status': 'error',
                'message': f'Error processing visual analysis: {str(db_error)}',
                'video_id': video_id,
                'progress': 100
            }, to=sid)
            
            # Send empty data
            await sio.emit('visual_analysis_data', {
                'status': 'error',
                'video_id': video_id,
                'error': str(db_error),
                'data': {
                    'frames': [],
                    'scenes': [],
                    'topics': [],
                    'highlights': []
                }
            }, to=sid)
            
    except Exception as e:
        logger.error(f"Error in analyze_visual_youtube: {str(e)}", exc_info=True)
        await sio.emit('error', {'message': f'Error in visual analysis: {str(e)}'}, to=sid)
        
        # Send error status
        await sio.emit('visual_analysis_status', {
            'status': 'error',
            'message': f'Error processing visual analysis: {str(e)}',
            'video_id': video_id if 'video_id' in locals() else "unknown",
            'progress': 100
        }, to=sid)
        
        # Send empty data
        await sio.emit('visual_analysis_data', {
            'status': 'error',
            'video_id': video_id if 'video_id' in locals() else "unknown",
            'error': str(e),
            'data': {
                'frames': [],
                'scenes': [],
                'topics': [],
                'highlights': []
            }
        }, to=sid)

# NEW ROUTE HANDLER for the specific endpoint causing 404s
@sio.event
async def visual_analysis_youtube(sid, data):
    """Handle the visual-analysis/youtube_ pattern that's causing 404s"""
    logger.info(f"Received visual_analysis_youtube request, forwarding to analyze_visual_youtube")
    await analyze_visual_youtube(sid, data)

@sio.event
async def get_transcript_timestamps(sid, data):
    """
    Get word-level timestamps from transcript files
    
    Args:
        sid: Socket ID
        data: Dictionary with video_id
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received transcript timestamps request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        
        if not video_id:
            await sio.emit('error', {'message': 'Missing video_id'}, to=sid)
            return
        
        # Get standardized IDs
        _, upload_id, file_id = standardize_video_id(video_id)
        logger.info(f"Getting transcript timestamps for video_id: {video_id} (upload_id: {upload_id}, file_id: {file_id})")
        
        # Try with all ID variants
        id_variants = [video_id, upload_id, file_id]
        timestamps_data = None
        
        for id_variant in id_variants:
            if not id_variant:
                continue
                
            try:
                # Try getting timestamps for this ID variant
                timestamps_variant = await get_timestamps_for_video(id_variant)
                
                if timestamps_variant and timestamps_variant.get("formatted_timestamps"):
                    timestamps_data = timestamps_variant
                    logger.info(f"Found timestamps using ID variant: {id_variant}")
                    break
            except Exception as variant_error:
                logger.warning(f"Error getting timestamps for variant {id_variant}: {variant_error}")
        
        # If still no timestamps, try generating from transcript
        if not timestamps_data or not timestamps_data.get("formatted_timestamps"):
            logger.warning(f"No timestamps found for any ID variant, generating from transcript")
            
            # Try loading transcript with each ID variant
            transcript_text = None
            for id_variant in id_variants:
                if not id_variant:
                    continue
                    
                variant_text = await load_transcription(id_variant)
                if variant_text and variant_text.strip():
                    transcript_text = variant_text
                    logger.info(f"Found transcript using ID variant: {id_variant}")
                    break
            
            if not transcript_text or transcript_text.strip() == "":
                logger.error(f"No transcript text available for generating timestamps")
                await sio.emit('transcript_timestamps', {
                    'status': 'warning',
                    'video_id': video_id,
                    'upload_id': upload_id,
                    'file_id': file_id,
                    'timestamps': [],
                    'message': 'No transcript available to generate timestamps'
                }, to=sid)
                return
            
            # Generate timestamps from transcript text
            timestamps_data = await generate_fallback_timestamps(transcript_text, upload_id or video_id)
        
        # Send the formatted timestamps to the client
        formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
        
        await sio.emit('transcript_timestamps', {
            'status': 'success',
            'video_id': video_id,
            'upload_id': upload_id,
            'file_id': file_id,
            'timestamps': formatted_timestamps
        }, to=sid)
        
        logger.info(f"Sent {len(formatted_timestamps)} transcript timestamps for video_id: {video_id}")
        
    except Exception as e:
        error_msg = f"Error in get_transcript_timestamps handler: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await sio.emit('error', {'message': error_msg}, to=sid)
        
        # Send empty timestamps array as fallback
        await sio.emit('transcript_timestamps', {
            'status': 'error',
            'video_id': video_id if 'video_id' in locals() else 'unknown',
            'timestamps': [],
            'error': str(e)
        }, to=sid)

@sio.event
async def get_video_summary(sid, data):
    """
    Generate and send a summary of the video
    
    Args:
        sid: Socket ID
        data: Dictionary with video_id
    """
    client_id = session_data.get(sid, {}).get('client_id', sid)
    logger.info(f"Received summary request from client {client_id}")
    
    try:
        # Validate input data
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
            
        video_id = data.get('video_id')
        
        if not video_id:
            await sio.emit('error', {'message': 'Missing video_id'}, to=sid)
            return
        
        # Get standardized IDs
        _, upload_id, file_id = standardize_video_id(video_id)
        
        # Let the client know we're generating a summary
        await sio.emit('ai_thinking', {
            'status': 'thinking',
            'video_id': video_id,
            'message': 'Generating summary...'
        }, to=sid)
        
        # Generate summary using AI service
        try:
            from app.services.ai import summarize_video
            # Try each ID variant
            summary = None
            for id_variant in [video_id, upload_id, file_id]:
                if not id_variant:
                    continue
                    
                try:
                    variant_summary = await summarize_video(id_variant)
                    if variant_summary:
                        summary = variant_summary
                        logger.info(f"Generated summary using ID variant: {id_variant}")
                        break
                except Exception as variant_error:
                    logger.warning(f"Failed to summarize with ID variant {id_variant}: {variant_error}")
            
            # If still no summary, create a basic one
            if not summary:
                # Try to load transcript with each ID variant
                transcript_text = None
                for id_variant in [video_id, upload_id, file_id]:
                    if not id_variant:
                        continue
                        
                    variant_text = await load_transcription(id_variant)
                    if variant_text and variant_text.strip():
                        transcript_text = variant_text
                        logger.info(f"Found transcript using ID variant: {id_variant}")
                        break
                
                if transcript_text:
                    if len(transcript_text) > 500:
                        summary = f"Here's a brief excerpt of the content:\n\n{transcript_text[:500]}...\n\n(AI summary generation is unavailable)"
                    else:
                        summary = f"Transcript:\n\n{transcript_text}\n\n(AI summary generation is unavailable)"
                else:
                    summary = "No transcript available to generate a summary."
        except ImportError:
            logger.error("ai.py module not found or missing summarize_video function")
            # Fallback simple summary
            transcript_text = await load_transcription(upload_id or video_id)
            if not transcript_text:
                summary = "No transcript available to generate a summary."
            elif len(transcript_text) > 500:
                summary = f"Here's a brief excerpt of the content:\n\n{transcript_text[:500]}...\n\n(AI summary generation is unavailable)"
            else:
                summary = f"Transcript:\n\n{transcript_text}\n\n(AI summary generation is unavailable)"
        
        # Format as AI response
        await sio.emit('ai_response', {
            'status': 'success',
            'video_id': video_id,
            'answer': summary,
            'timestamps': [],
            'is_complete': True  # Add completion flag
        }, to=sid)
        
    except Exception as e:
        error_msg = f"Error in get_video_summary handler: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await sio.emit('error', {'message': error_msg}, to=sid)
        
        # Send error response with completion flag
        await sio.emit('ai_response', {
            'status': 'error',
            'message': f'Error generating summary: {str(e)}',
            'video_id': video_id if 'video_id' in locals() else None,
            'answer': "An error occurred while generating the summary.",
            'is_complete': True
        }, to=sid)

# ADDED SPECIAL HANDLER FOR VISUAL ANALYSIS REQUEST FORMAT THAT'S CAUSING 404s
@sio.event
async def visual_analysis(sid, data):
    """Handle visual analysis requests - route to the proper handler"""
    logger.info(f"Received visual_analysis request, forwarding to analyze_visual_youtube")
    await analyze_visual_youtube(sid, data)

@sio.event
async def get_video_timestamps(sid, data):
    """Get timestamps for a video from database to display in the sidebar"""
    try:
        video_id = data.get('video_id')
        if not video_id:
            await sio.emit('error', {
                'message': 'Missing video_id parameter'
            }, to=sid)
            return
        
        # Get standardized IDs
        _, upload_id, file_id = standardize_video_id(video_id)
        
        # Try creating the table if it doesn't exist first
        from app.utils.database import get_db_context
        from sqlalchemy import text
        
        with get_db_context() as db:
            # Create the table if it doesn't exist
            db.execute(text("""
            CREATE TABLE IF NOT EXISTS video_timestamps (
                id SERIAL PRIMARY KEY,
                video_id TEXT NOT NULL,
                timestamp FLOAT NOT NULL,
                formatted_time TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """))
            db.commit()
            
            # Try to find timestamps for any ID variant
            timestamps = None
            for id_variant in [video_id, upload_id, file_id]:
                if not id_variant:
                    continue
                    
                # Check if we have any timestamps for this variant
                result = db.execute(
                    text("""
                    SELECT timestamp, formatted_time, description 
                    FROM video_timestamps 
                    WHERE video_id = :video_id
                    ORDER BY timestamp
                    """),
                    {"video_id": str(id_variant)}
                )
                
                variant_timestamps = [
                    {
                        "time": row[0],
                        "time_formatted": row[1],
                        "text": row[2]
                    }
                    for row in result
                ]
                
                if variant_timestamps:
                    timestamps = variant_timestamps
                    logger.info(f"Found {len(timestamps)} timestamps in DB for variant {id_variant}")
                    break
            
            # If no timestamps in DB, try the file-based approach
            if not timestamps:
                # Try file-based approach for each ID variant
                for id_variant in [video_id, upload_id, file_id]:
                    if not id_variant:
                        continue
                        
                    timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{id_variant}_timestamps.json")
                    if os.path.exists(timestamp_path):
                        try:
                            with open(timestamp_path, "r") as f:
                                timestamps_data = json.load(f)
                                formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
                                
                                # Convert to expected format
                                variant_timestamps = [
                                    {
                                        "time": ts.get("start_time", 0),
                                        "time_formatted": ts.get("display_time", "00:00"),
                                        "text": ts.get("text", "")
                                    }
                                    for ts in formatted_timestamps
                                ]
                                
                                if variant_timestamps:
                                    timestamps = variant_timestamps
                                    logger.info(f"Loaded {len(timestamps)} timestamps from file for variant {id_variant}")
                                    break
                        except Exception as file_error:
                            logger.error(f"Error loading timestamps from file: {str(file_error)}")
            
            # If still no timestamps, generate fallback
            if not timestamps:
                # Try to generate from transcript
                transcript_text = None
                
                # Try each ID variant
                for id_variant in [video_id, upload_id, file_id]:
                    if not id_variant:
                        continue
                        
                    variant_text = await load_transcription(id_variant)
                    if variant_text and variant_text.strip():
                        transcript_text = variant_text
                        logger.info(f"Found transcript using ID variant: {id_variant}")
                        break
                
                if transcript_text and transcript_text.strip():
                    timestamps_data = await generate_fallback_timestamps(transcript_text, upload_id or video_id)
                    formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
                    
                    # Convert to expected format
                    timestamps = [
                        {
                            "time": ts.get("start_time", 0),
                            "time_formatted": ts.get("display_time", "00:00"),
                            "text": ts.get("text", "")
                        }
                        for ts in formatted_timestamps
                    ]
                    logger.info(f"Generated {len(timestamps)} fallback timestamps")
            
            # If still no timestamps, use empty array
            if not timestamps:
                timestamps = []
            
            await sio.emit('video_timestamps', {
                'status': 'success',
                'timestamps': timestamps,
                'video_id': video_id,
                'upload_id': upload_id,
                'file_id': file_id
            }, to=sid)
            
    except Exception as e:
        logger.error(f"Error getting timestamps via socket: {e}")
        await sio.emit('error', {
            'message': f'Error getting timestamps: {str(e)}'
        }, to=sid)
        
        # Still send empty timestamp array to avoid frontend errors
        await sio.emit('video_timestamps', {
            'status': 'error',
            'timestamps': [],
            'video_id': video_id if 'video_id' in locals() else 'unknown'
        }, to=sid)

# New function to handle heartbeat and keep connections alive
@sio.event
async def heartbeat(sid, data=None):
    """Handle heartbeat requests to prevent timeouts"""
    try:
        await sio.emit('heartbeat', {'timestamp': time.time()}, to=sid)
    except Exception as e:
        logger.error(f"Error sending heartbeat: {str(e)}")

# FIXED: Add the setup_socketio function to main setup
def setup_socketio(app):
    """Set up Socket.IO with FastAPI app"""
    # Mount at exactly /socket.io
    app.mount("/socket.io", socket_app)
    logger.info("Socket.IO mounted at /socket.io")
    
    # Add REST API routes to prevent 404 errors
    setup_fastapi_routes(app)
    
    return sio