# app/api/videos.py
"""
Video management API endpoints for Luna AI
Handles video upload, processing, retrieval and analysis
Including visual analysis functionality
"""
from typing import Any, List, Dict, Optional
from uuid import UUID
import os
import re
import uuid
import json
import shutil
import tempfile
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
# Import from our simplified structure
from app.utils.database import get_db
from app.services.auth import get_current_user
from app.services.video_processing import process_video, process_youtube_video
from app.services.ai import generate_ai_response
from app.services.object_detection import detect_objects_in_video
from app.services.visual_analysis import get_visual_analysis_service
from app.models.user import User
from app.models.video import Video
from app.config import settings
from app.dependencies import get_visual_analysis_service_dependency
# Import the transcription functions directly
from app.services.transcription import transcribe_uploaded_file, process_audio_file
# Change prefix to match expected API structure
router = APIRouter(prefix="/api/v1/videos")
logger = logging.getLogger("videos_api")

@router.get("/", response_model=List[Dict[str, Any]])
def list_videos(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve videos for the current user
    Returns list of videos with pagination
    """
    from sqlalchemy import text
    
    # Get videos with direct SQL query
    result = db.execute(
        text("""
        SELECT id, video_name, video_url, processed, created_at, updated_at
        FROM videos
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :skip
        """),
        {
            "user_id": current_user.id,
            "limit": limit,
            "skip": skip
        }
    )
    
    videos = []
    for row in result:
        videos.append({
            "id": row[0],
            "title": row[1],
            "url": row[2],
            "processed": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        })
    
    return videos

@router.post("/upload", response_model=Dict[str, Any])
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    tab_id: str = Form(...),
    video_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Upload and process a video file
    Saves file to disk and starts background processing
    """
    # Validate file type
    valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    
    if file_ext not in valid_video_extensions:
        logger.warning(f"Invalid video file type: {file_ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video file type. Supported types: {', '.join(valid_video_extensions)}"
        )
    
    # Generate or use provided video_id
    if not video_id:
        video_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated new video_id: {video_id}")
    
    # Generate a unique filename that includes the video_id
    unique_filename = f"{uuid.uuid4()}_{video_id}{file_ext}"
    
    # Make sure upload directory exists as a string path
    upload_dir = str(settings.UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, unique_filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            # Use a buffer to handle large files efficiently
            shutil.copyfileobj(video_file.file, buffer)
        
        # Check if file was actually saved
        if not os.path.exists(file_path):
            raise Exception(f"File was not saved to {file_path}")
            
        logger.info(f"Successfully saved file to {file_path}, size: {os.path.getsize(file_path)} bytes")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    
    # Save video information to the database
    from sqlalchemy import text
    from datetime import datetime
    timestamp = datetime.utcnow().isoformat()
    
    try:
        # Insert with direct SQL
        db.execute(
            text("""
            INSERT INTO videos 
            (id, video_name, video_url, user_id, processed, created_at, updated_at, tab_id, is_audio) 
            VALUES (:id, :video_name, :video_url, :user_id, :processed, :created_at, :updated_at, :tab_id, :is_audio)
            """),
            {
                "id": video_id,
                "video_name": video_file.filename[:255],
                "video_url": file_path,  # Use string path
                "user_id": current_user.id,
                "processed": False,
                "created_at": timestamp,
                "updated_at": timestamp,
                "tab_id": tab_id,
                "is_audio": False  # Mark as video file
            }
        )
        db.commit()
    except Exception as db_error:
        # Clean up the file if database operation fails
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        logger.error(f"Database error saving video: {str(db_error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(db_error)}"
        )
    
    # Define output file for transcription explicitly with video_id
    transcription_dir = str(settings.TRANSCRIPTION_DIR)
    os.makedirs(transcription_dir, exist_ok=True)
    output_file = os.path.join(transcription_dir, f"transcription_{video_id}.json")
    
    # Notify about upload via WebSockets if socket.io is available
    try:
        # Try both possible imports for Socket.IO
        try:
            from app.api.websockets import sio
            sio_source = "app.api.websockets"
        except ImportError:
            try:
                from app.socketio_server import sio
                sio_source = "app.socketio_server"
            except ImportError:
                sio = None
                sio_source = "none"
        
        if sio:
            await sio.emit('transcription_status', {
                'status': 'received',
                'message': 'Video file received, starting processing...',
                'video_id': video_id,
                'progress': 5
            }, room=tab_id)
            logger.info(f"Sent 'received' status notification using {sio_source}")
    except Exception as e:
        logger.error(f"Error sending WebSocket notification: {str(e)}")
    
    # Start the transcription process directly
    try:
        # IMPORTANT FIX: Use video_id parameter, not file_id
        background_tasks.add_task(
            process_audio_file,
            file_path=file_path,
            video_id=video_id,  # Correct parameter name
            language_code="en",
            tab_id=tab_id,
            session_id=tab_id,  # Add session_id for WebSocket notifications
            output_file=output_file,
            debug_mode=False  # Explicitly set debug mode to false
        )
        
        logger.info(f"Background task started for video transcription: {video_id}")
    except Exception as e:
        logger.error(f"Error starting transcription process: {str(e)}", exc_info=True)
        # Notify clients about error
        try:
            # Try both possible imports again for better compatibility
            try:
                from app.api.websockets import sio
            except ImportError:
                from app.socketio_server import sio
                
            await sio.emit('transcription_status', {
                'status': 'error',
                'message': f'Error starting transcription: {str(e)}',
                'video_id': video_id,
                'progress': 0
            }, room=tab_id)
        except Exception as notify_error:
            logger.error(f"Error sending error notification: {str(notify_error)}")
    
    return {
        "message": "Video upload completed, processing started", 
        "video_id": video_id,
        "success": True
    }

@router.post("/audio/upload", response_model=Dict[str, Any])
async def upload_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    tab_id: str = Form(...),
    audio_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Upload and process an audio file (MP3, WAV, etc.) for transcription
    """
    # Validate file type
    valid_audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    if file_ext not in valid_audio_extensions:
        logger.warning(f"Invalid audio file type: {file_ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio file type. Supported types: {', '.join(valid_audio_extensions)}"
        )
    
    # Generate or use provided file_id
    if not audio_id:
        audio_id = f"audio_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated new audio_id: {audio_id}")
    
    # Generate a unique filename that includes the file_id
    unique_filename = f"{uuid.uuid4()}_{audio_id}{file_ext}"
    
    # Create upload directory as a string path
    upload_dir = str(settings.UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save the uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        logger.info(f"Successfully saved file to {file_path}, size: {os.path.getsize(file_path)} bytes")    
    except Exception as e:
        logger.error(f"Failed to save uploaded audio file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded audio file: {str(e)}"
        )
    
    # Save audio information to the database
    from sqlalchemy import text
    from datetime import datetime
    timestamp = datetime.utcnow().isoformat()
    
    try:
        # Insert with direct SQL - store as a video type with audio flag
        db.execute(
            text("""
            INSERT INTO videos 
            (id, video_name, video_url, user_id, processed, created_at, updated_at, tab_id, is_audio) 
            VALUES (:id, :video_name, :video_url, :user_id, :processed, :created_at, :updated_at, :tab_id, :is_audio)
            """),
            {
                "id": audio_id,
                "video_name": audio_file.filename[:255],
                "video_url": file_path,  # Use string path
                "user_id": current_user.id,
                "processed": False,
                "created_at": timestamp,
                "updated_at": timestamp,
                "tab_id": tab_id,
                "is_audio": True  # Mark as audio file
            }
        )
        db.commit()
    except Exception as db_error:
        # Clean up the file if database operation fails
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        logger.error(f"Database error saving audio: {str(db_error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(db_error)}"
        )
    
    # Define output file for transcription
    transcription_dir = str(settings.TRANSCRIPTION_DIR)
    os.makedirs(transcription_dir, exist_ok=True)
    output_file = os.path.join(transcription_dir, f"transcription_{audio_id}.json")
    
    # Notify about upload via WebSockets
    try:
        # Try both possible imports for Socket.IO
        try:
            from app.api.websockets import sio
            sio_source = "app.api.websockets"
        except ImportError:
            try:
                from app.socketio_server import sio
                sio_source = "app.socketio_server"
            except ImportError:
                sio = None
                sio_source = "none"
                
        if sio:
            await sio.emit('transcription_status', {
                'status': 'received',
                'message': 'Audio file received, starting processing...',
                'video_id': audio_id,  # Use video_id for consistency with frontend
                'progress': 5
            }, room=tab_id)
            logger.info(f"Sent 'received' status notification using {sio_source}")
    except Exception as e:
        logger.error(f"Error sending status update: {str(e)}")
    
    # Start the real transcription processing
    try:
        # IMPORTANT FIX: Use video_id parameter, not file_id
        background_tasks.add_task(
            process_audio_file,
            file_path=file_path,
            video_id=audio_id,  # Correct parameter name
            language_code="en",
            tab_id=tab_id,
            session_id=tab_id,  # Add session_id for WebSocket notifications
            output_file=output_file,
            file_type="audio",  # Specify file type as audio
            debug_mode=False  # Explicitly set debug mode to false
        )
        
        logger.info(f"Background task started for audio processing: {audio_id}")
    except Exception as e:
        logger.error(f"Error starting audio transcription: {str(e)}", exc_info=True)
        # Notify clients about error
        try:
            # Try both possible imports again for better compatibility
            try:
                from app.api.websockets import sio
            except ImportError:
                from app.socketio_server import sio
                
            await sio.emit('transcription_status', {
                'status': 'error',
                'message': f'Error starting transcription: {str(e)}',
                'video_id': audio_id,
                'progress': 0
            }, room=tab_id)
        except Exception as notify_error:
            logger.error(f"Error sending error notification: {str(notify_error)}")
    
    return {
        "message": "Audio upload started, processing in background",
        "audio_id": audio_id,
        "video_id": audio_id,  # Include video_id for frontend compatibility
        "success": True
    }

# For backward compatibility - keep the original audio upload endpoint but redirect to the new one
@router.post("/upload/audio", response_model=Dict[str, Any])
async def upload_audio_legacy(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    tab_id: str = Form(...),
    file_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Legacy endpoint for audio uploads - redirects to the new endpoint"""
    logger.info(f"Using legacy audio upload endpoint with file_id={file_id}, tab_id={tab_id}")
    
    # Create a mock request
    class MockRequest:
        pass
    mock_request = MockRequest()
    
    # Properly map file_id to audio_id before calling upload_audio
    return await upload_audio(
        request=mock_request,
        background_tasks=background_tasks,
        audio_file=audio_file,
        tab_id=tab_id,
        audio_id=file_id,  # Pass file_id as audio_id
        current_user=current_user,
        db=db
    )

# Modified to accept POST requests for better handling of URL data
# Modify the process_youtube_video_endpoint function (around line 450)

@router.post("/youtube", response_model=Dict[str, Any])
async def process_youtube_video_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Process a YouTube video URL
    Extracts info and starts background processing
    """
    try:
        # Parse request body
        data = await request.json()
        youtube_url = data.get("youtube_url") or data.get("url")  # Support both parameter names
        tab_id = data.get("tab_id")
        video_id = data.get("video_id")  # Optional video_id from request
        
        # Validate required fields
        if not youtube_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="YouTube URL is required"
            )
            
        if not tab_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tab ID is required"
            )
            
        logger.info(f"Processing YouTube URL: {youtube_url}, Tab ID: {tab_id}")
        
        # Use the new extraction function to clean the URL and get video ID
        extracted_video_id, clean_youtube_url = extract_clean_youtube_video_id(youtube_url)
        
        if not extracted_video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)"
            )
            
        # Use the clean URL for further processing
        youtube_url = clean_youtube_url
        logger.info(f"Using clean YouTube URL: {youtube_url}")
            
        try:
            import yt_dlp as youtube_dl
        except ImportError:
            try:
                import youtube_dl
            except ImportError:
                logger.error("Neither yt-dlp nor youtube-dl is installed")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="YouTube download libraries not available"
                )
        
        # yt-dlp configuration
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
            },
        }
        
        # Extract info without downloading
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)
                video_title = info_dict.get('title', 'Untitled Video')
        except Exception as extract_error:
            logger.error(f"Error extracting YouTube info: {str(extract_error)}")
            video_title = "YouTube Video"  # Fallback title
        
        # Generate video ID if not provided
        if not video_id:
            video_id = f"youtube_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generated new video_id: {video_id}")
        
        # Save video information to database
        from sqlalchemy import text
        from datetime import datetime
        
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Insert with direct SQL
            db.execute(
                text("""
                INSERT INTO videos 
                (id, video_name, video_url, user_id, processed, created_at, updated_at, tab_id) 
                VALUES (:id, :video_name, :video_url, :user_id, :processed, :created_at, :updated_at, :tab_id)
                """),
                {
                    "id": video_id,
                    "video_name": video_title[:255],
                    "video_url": youtube_url,
                    "user_id": current_user.id,
                    "processed": False,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "tab_id": tab_id  # Add tab_id for YouTube videos too
                }
            )
            db.commit()
        except Exception as db_error:
            logger.error(f"Database error saving YouTube video: {str(db_error)}")
            # Continue anyway since the process can still work without the DB entry
        
        # Define output file for transcription with video_id
        output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
        
        # Ensure transcription directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Send initial notification
        try:
            from app.socketio_server import sio
            await sio.emit('transcription_status', {
                'status': 'received',
                'message': 'YouTube URL received, starting processing...',
                'video_id': video_id,
                'progress': 5
            }, room=tab_id)
        except Exception as notify_error:
            logger.error(f"Error sending initial status: {str(notify_error)}")
        
        # Process the YouTube video in the background
        try:
            # Use process_youtube_video from transcription_service
            logger.info(f"Starting background task for YouTube processing: {video_id}")
            
            background_tasks.add_task(
                process_youtube_video,
                youtube_url=youtube_url,
                video_id=video_id,
                tab_id=tab_id,
                session_id=tab_id,  # Pass tab_id as session_id for WebSocket notifications
                language="en"
            )
            
            logger.info(f"Background task started for YouTube processing: {video_id}")
        except Exception as task_error:
            logger.error(f"Error starting background task: {str(task_error)}")
            # Notify clients about error
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': f'Error starting YouTube processing: {str(task_error)}',
                    'video_id': video_id,
                    'progress': 0
                }, room=tab_id)
            except Exception:
                pass
        
        return {
            "message": "YouTube processing started", 
            "video_id": video_id,
            "success": True
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions with their original status codes
        raise
    except Exception as e:
        logger.error(f"Failed to process YouTube URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process YouTube URL: {str(e)}"
        )
# Add this improved extraction function to videos.py (around line 500)

def extract_clean_youtube_video_id(url: str) -> tuple:
    """
    Extract video ID from YouTube URL and return both ID and clean URL
    Handles URLs with additional parameters correctly
    """
    try:
        from urllib.parse import urlparse, parse_qs
        
        # Create a clean URL to return
        clean_url = url
        video_id = None
        
        parsed_url = urlparse(url)
        
        # For youtube.com/watch URLs
        if parsed_url.netloc in ['youtube.com', 'www.youtube.com'] and parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                video_id = query_params['v'][0]
                # Create a clean URL with just the video ID
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
                
        # For youtu.be URLs
        elif parsed_url.netloc == 'youtu.be':
            video_id = parsed_url.path.lstrip('/').split('/')[0]
            # Create a clean URL
            clean_url = f"https://youtu.be/{video_id}"
            
        # For youtube.com/shorts URLs
        elif parsed_url.netloc in ['youtube.com', 'www.youtube.com'] and '/shorts/' in parsed_url.path:
            parts = parsed_url.path.split('/shorts/')
            if len(parts) > 1:
                video_id = parts[1].split('/')[0]
                # Create a clean URL
                clean_url = f"https://www.youtube.com/shorts/{video_id}"
                
        # For youtube.com/embed URLs
        elif parsed_url.netloc in ['youtube.com', 'www.youtube.com'] and '/embed/' in parsed_url.path:
            parts = parsed_url.path.split('/embed/')
            if len(parts) > 1:
                video_id = parts[1].split('/')[0]
                # Create a clean URL
                clean_url = f"https://www.youtube.com/embed/{video_id}"
                
        logger.info(f"Extracted YouTube ID {video_id} from URL: {url}")
        return video_id, clean_url
        
    except Exception as e:
        logger.error(f"Error extracting YouTube video ID: {str(e)}")
        return None, url
    
# Keep the GET endpoint for compatibility, but make it call the POST handler
@router.get("/youtube", response_model=Dict[str, Any])
async def process_youtube_url_get(
    youtube_url: str,
    tab_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Process a YouTube video URL (GET method for backward compatibility)
    Forwards to the POST handler
    """
    # Create a simple JSON-like object to pass to the POST handler
    class MockRequest:
        async def json(self):
            return {"youtube_url": youtube_url, "tab_id": tab_id}
    
    # Call the POST handler with the mock request
    return await process_youtube_video_endpoint(
        request=MockRequest(),
        background_tasks=background_tasks,
        current_user=current_user,
        db=db
    )

@router.get("/transcription", response_model=Dict[str, str])
async def get_transcription(
    tab_id: Optional[str] = None,
    video_id: Optional[str] = None,  # Added video_id parameter
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get the transcription for a video using tab_id or video_id
    First tries to get from file, then falls back to database
    """
    if not tab_id and not video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either Tab ID or Video ID is required",
        )
    
    from sqlalchemy import text
    
    # If video_id is provided, use it directly
    if video_id:
        target_video_id = video_id
    else:
        # Try to find latest video processed for this tab
        result = db.execute(
            text("""
            SELECT id FROM videos
            WHERE user_id = :user_id AND tab_id = :tab_id
            ORDER BY created_at DESC
            LIMIT 1
            """),
            {"user_id": current_user.id, "tab_id": tab_id}
        )
        
        video_row = result.fetchone()
        
        if not video_row:
            # Return empty transcription if no videos found
            return {"transcription": ""}
        
        target_video_id = video_row[0]
    
    # Verify user has access to this video
    result = db.execute(
        text("SELECT user_id FROM videos WHERE id = :id"),
        {"id": target_video_id}
    )
    video_access = result.fetchone()
    
    if not video_access or video_access[0] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    # Try direct transcription file first
    transcription_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{target_video_id}.json")
    if os.path.exists(transcription_file):
        try:
            with open(transcription_file, "r") as f:
                transcription_data = json.load(f)
                transcription = transcription_data.get("text", "")
                return {"transcription": transcription}
        except Exception as e:
            logger.error(f"Error reading transcription file: {e}")
    
    # Try to get transcription data from video data file
    if tab_id:
        data_file = os.path.join(settings.TRANSCRIPTION_DIR, f"video_data_{target_video_id}_{tab_id}.json")
        
        if os.path.exists(data_file):
            try:
                with open(data_file, "r") as f:
                    video_data = json.load(f)
                
                transcription_filename = video_data.get("transcription_filename")
                if transcription_filename and os.path.exists(transcription_filename):
                    with open(transcription_filename, "r") as f:
                        transcription_data = json.load(f)
                        transcription = transcription_data.get("text", "")
                    return {"transcription": transcription}
            except Exception as e:
                logger.error(f"Error reading video data file: {e}")
    
    # Fall back to database transcription
    result = db.execute(
        text("SELECT transcription FROM videos WHERE id = :id"),
        {"id": target_video_id}
    )
    video_data = result.fetchone()
    
    return {"transcription": video_data[0] if video_data and video_data[0] else ""}

@router.get("/{video_id}/transcription", response_model=Dict[str, str])
async def get_video_transcription(
    video_id: str,
    tab_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get the transcription for a specific video
    First tries to get from file, then falls back to database
    """
    from sqlalchemy import text
    
    # Verify video exists and belongs to user
    result = db.execute(
        text("""
        SELECT id, transcription, user_id 
        FROM videos 
        WHERE id = :id
        """),
        {"id": str(video_id)}
    )
    
    video_row = result.fetchone()
    
    if not video_row or video_row[2] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Try direct transcription file first
    transcription_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
    if os.path.exists(transcription_file):
        try:
            with open(transcription_file, "r") as f:
                transcription_data = json.load(f)
                transcription = transcription_data.get("text", "")
                return {"transcription": transcription}
        except Exception as e:
            logger.error(f"Error reading transcription file: {e}")
    
    # Try to get transcription data from video data file
    if tab_id:
        data_file = os.path.join(settings.TRANSCRIPTION_DIR, f"video_data_{video_id}_{tab_id}.json")
        
        if os.path.exists(data_file):
            try:
                with open(data_file, "r") as f:
                    video_data = json.load(f)
                
                transcription_filename = video_data.get("transcription_filename")
                if transcription_filename and os.path.exists(transcription_filename):
                    with open(transcription_filename, "r") as f:
                        transcription_data = json.load(f)
                        transcription = transcription_data.get("text", "")
                    return {"transcription": transcription}
            except Exception as e:
                logger.error(f"Error reading video data file: {e}")
    
    # Fall back to database transcription
    return {"transcription": video_row[1] or ""}

@router.post("/{video_id}/detect-object", response_model=List[Dict[str, Any]])
async def detect_object(
    video_id: UUID,
    target_object: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Detect a specific object in a video
    Returns list of timestamps where the object appears
    """
    from sqlalchemy import text
    
    # Get video URL
    result = db.execute(
        text("SELECT video_url, user_id FROM videos WHERE id = :id"),
        {"id": str(video_id)}
    )
    video_row = result.fetchone()
    
    if not video_row or video_row[1] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    video_url = video_row[0]
    
    try:
        # Call object detection service
        results = await detect_objects_in_video(video_url, [target_object])
        
        if results:
            return [{"timestamp": float(timestamp), "label": label} for timestamp, label, _ in results]
        else:
            return []
            
    except Exception as e:
        logger.error(f"Failed to detect object: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect object: {str(e)}"
        )

@router.post("/{video_id}/ask", response_model=Dict[str, Any])
async def ask_ai(
    video_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Ask AI a question about the video
    Uses video content to generate a contextual answer
    """
    from sqlalchemy import text
    
    # Verify video ownership
    result = db.execute(
        text("SELECT id, user_id FROM videos WHERE id = :id"),
        {"id": str(video_id)}
    )
    video_row = result.fetchone()
    
    if not video_row or video_row[1] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Get question from request body
    data = await request.json()
    question = data.get("question")
    tab_id = data.get("tab_id")
    
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required",
        )
    
    try:
        # Generate AI response
        response = await generate_ai_response(
            video_id=str(video_id),
            question=question,
            tab_id=tab_id
        )
        return response
    except Exception as e:
        logger.error(f"Failed to generate AI response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate AI response: {str(e)}"
        )

@router.delete("/{video_id}")
async def delete_video(
    video_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Delete a video
    Removes database record and associated files
    """
    from sqlalchemy import text
    
    # Check if video exists and belongs to user
    result = db.execute(
        text("SELECT id, video_url, user_id FROM videos WHERE id = :id"),
        {"id": str(video_id)}
    )
    video_row = result.fetchone()
    
    if not video_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check ownership
    if video_row[2] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    video_url = video_row[1]
    
    # Delete timestamps associated with this video (new!)
    try:
        db.execute(
            text("DELETE FROM video_timestamps WHERE video_id = :video_id"),
            {"video_id": str(video_id)}
        )
    except Exception as e:
        # This might fail if the table doesn't exist yet, which is fine
        logger.warning(f"Could not delete timestamps for video {video_id}: {e}")
    
    # Delete video record
    db.execute(
        text("DELETE FROM videos WHERE id = :id"),
        {"id": str(video_id)}
    )
    
    # Delete associated frames
    try:
        db.execute(
            text("DELETE FROM frames WHERE video_id = :video_id"),
            {"video_id": str(video_id)}
        )
    except Exception as e:
        # This might fail if the table doesn't exist yet, which is fine
        logger.warning(f"Could not delete frames for video {video_id}: {e}")
    
    db.commit()
    
    # Delete files if they exist
    try:
        if os.path.exists(video_url) and str(settings.UPLOAD_DIR) in video_url:
            os.remove(video_url)
        
        # Delete frames directory
        frames_dir = settings.FRAME_DIR / f"video_{video_id}"
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        
        # Delete transcription data and files
        for file in settings.TRANSCRIPTION_DIR.glob(f"transcription_{video_id}.json"):
            os.remove(file)
        
        for file in settings.TRANSCRIPTION_DIR.glob(f"video_data_{video_id}_*.json"):
            os.remove(file)
            
        for file in settings.TRANSCRIPTION_DIR.glob(f"visual_data_{video_id}.json"):
            os.remove(file)
            
        # Delete timestamp files
        timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
        if os.path.exists(timestamp_path):
            os.remove(timestamp_path)
    except Exception as e:
        # Log but don't fail if file deletion fails
        logger.error(f"Error deleting files for video {video_id}: {str(e)}")
    
    return {"message": "Video deleted successfully"}

#-------------------------------------------------
# Visual Analysis Endpoints
#-------------------------------------------------
@router.post("/{video_id}/analyze-visual")
async def analyze_video_visual(
    video_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    visual_service = Depends(get_visual_analysis_service_dependency)
):
    """
    Analyze the visual content of a video
    This will extract frames, detect objects, and generate scene descriptions
    """
    try:
        # Verify video exists and user has access
        from sqlalchemy import text
        result = db.execute(
            text("SELECT video_url, user_id FROM videos WHERE id::text = :id"),
            {"id": str(video_id)}
        )
        video_row = result.fetchone()
        
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if str(video_row[1]) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        video_path = video_row[0]
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Start visual analysis in the background
        background_tasks.add_task(
            visual_service.process_video,
            video_path=video_path,
            video_id=video_id
        )
        
        return {
            "message": "Visual analysis started",
            "video_id": video_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting visual analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting visual analysis: {str(e)}")

@router.get("/{video_id}/visual-data")
async def get_visual_data(
    video_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    visual_service = Depends(get_visual_analysis_service_dependency)
):
    """
    Get visual analysis data for a video
    Returns scenes, frames, and summary information
    """
    try:
        # Verify video exists and user has access
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id FROM videos WHERE id::text = :id"),
            {"id": str(video_id)}
        )
        video_row = result.fetchone()
        
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if str(video_row[1]) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load visual data
        visual_data = await visual_service.load_visual_data(video_id)
        
        if not visual_data or "video_id" not in visual_data:
            return {
                "message": "No visual data available",
                "video_id": video_id,
                "status": "not_processed",
                "visual_summary": {
                    "overall_summary": "Video has not been analyzed visually"
                }
            }
        
        return visual_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visual data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting visual data: {str(e)}")

@router.post("/{video_id}/visual-question")
async def ask_visual_question(
    video_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    visual_service = Depends(get_visual_analysis_service_dependency)
):
    """
    Answer a question about the visual content of a video
    """
    try:
        # Verify video exists and user has access
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id FROM videos WHERE id::text = :id"),
            {"id": str(video_id)}
        )
        video_row = result.fetchone()
        
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if str(video_row[1]) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get question from request body
        data = await request.json()
        question = data.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Answer the question
        answer = await visual_service.answer_visual_question(video_id, question)
        
        return answer
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering visual question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering visual question: {str(e)}")

@router.get("/{video_id}/timestamp/{timestamp}")
async def get_content_at_timestamp(
    video_id: str,
    timestamp: float,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    visual_service = Depends(get_visual_analysis_service_dependency)
):
    """
    Get visual content at a specific timestamp
    """
    try:
        # Verify video exists and user has access
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id FROM videos WHERE id::text = :id"),
            {"id": str(video_id)}
        )
        video_row = result.fetchone()
        
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if str(video_row[1]) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load visual data
        visual_data = await visual_service.load_visual_data(video_id)
        
        if not visual_data or "frames" not in visual_data or not visual_data["frames"]:
            raise HTTPException(status_code=404, detail="No visual data available for this video")
        
        # Find the closest frame to the requested timestamp
        frames = visual_data["frames"]
        closest_frame = min(frames, key=lambda x: abs(x.get("timestamp", 0) - timestamp))
        
        # Find which scene this belongs to
        scenes = visual_data.get("scenes", [])
        containing_scene = None
        
        for scene in scenes:
            start_time = scene.get("start_time", 0)
            end_time = scene.get("end_time", 0)
            
            if start_time <= timestamp <= end_time:
                containing_scene = scene
                break
        
        result = {
            "timestamp": timestamp,
            "closest_frame": closest_frame,
            "containing_scene": containing_scene
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content at timestamp: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting content at timestamp: {str(e)}")

@router.get("/{video_id}/scenes")
async def get_video_scenes(
    video_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    visual_service = Depends(get_visual_analysis_service_dependency)
):
    """
    Get scene information for a video
    """
    try:
        # Verify video exists and user has access
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id FROM videos WHERE id::text = :id"),
            {"id": str(video_id)}
        )
        video_row = result.fetchone()
        
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if str(video_row[1]) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load visual data
        visual_data = await visual_service.load_visual_data(video_id)
        
        if not visual_data or "scenes" not in visual_data:
            return {
                "message": "No scene data available",
                "video_id": video_id,
                "scenes": []
            }
        
        return {
            "video_id": video_id,
            "scenes": visual_data["scenes"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video scenes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting video scenes: {str(e)}")

@router.get("/{video_id}/timestamps", response_model=List[Dict[str, Any]])
async def get_video_timestamps(
    video_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get timestamps for a video to display in the sidebar
    """
    try:
        # Verify video exists and user has access
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id FROM videos WHERE id::text = :id"),
            {"id": str(video_id)}
        )
        video_row = result.fetchone()
        
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if str(video_row[1]) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # First check for timestamps file in the dedicated directory
        timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
        # Ensure directory exists
        os.makedirs(os.path.dirname(timestamp_path), exist_ok=True)
        
        if os.path.exists(timestamp_path):
            try:
                with open(timestamp_path, "r") as f:
                    timestamp_data = json.load(f)
                    if "formatted_timestamps" in timestamp_data and timestamp_data["formatted_timestamps"]:
                        # Return the pre-formatted timestamps
                        return timestamp_data["formatted_timestamps"]
            except Exception as e:
                logger.error(f"Error reading timestamps file: {str(e)}")
                # Continue to database fallback
        
        try:
            # Create the timestamps table if it doesn't exist
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
            
            # Get timestamps from database
            result = db.execute(
                text("""
                SELECT timestamp, formatted_time, description 
                FROM video_timestamps 
                WHERE video_id = :video_id
                ORDER BY timestamp
                """),
                {"video_id": video_id}
            )
            
            timestamps = [
                {
                    "time": row[0],
                    "time_formatted": row[1],
                    "text": row[2]
                }
                for row in result
            ]
            
            # If we found timestamps in DB, let's also cache them to file
            if timestamps:
                try:
                    timestamp_data = {
                        "video_id": video_id,
                        "formatted_timestamps": timestamps
                    }
                    with open(timestamp_path, "w") as f:
                        json.dump(timestamp_data, f)
                except Exception as write_error:
                    logger.error(f"Error caching timestamps to file: {str(write_error)}")
            
            return timestamps
        except Exception as db_error:
            logger.error(f"Database error fetching timestamps: {str(db_error)}")
            
            # Generate timestamps from transcription if available
            try:
                # Try direct transcription file first
                transcription_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
                if os.path.exists(transcription_file):
                    with open(transcription_file, "r") as f:
                        transcription_data = json.load(f)
                        text = transcription_data.get("text", "")
                        
                        # Split text into sentences and generate timestamps
                        import re
                        from datetime import timedelta
                        
                        sentences = re.split(r'[.!?]+\s*', text)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        
                        # Estimate video duration (default to 5 minutes if we can't determine)
                        estimated_duration = 300  # 5 minutes in seconds
                        
                        # Generate evenly spaced timestamps
                        timestamps = []
                        for i, sentence in enumerate(sentences):
                            if i >= 13:  # Limit to 13 timestamps as seen in logs
                                break
                                
                            # Calculate timestamp
                            time_sec = (i / len(sentences)) * estimated_duration
                            time_obj = timedelta(seconds=time_sec)
                            formatted_time = str(time_obj).split('.')[0]  # Remove microseconds
                            
                            timestamps.append({
                                "time": time_sec,
                                "time_formatted": formatted_time,
                                "text": sentence[:50] + "..." if len(sentence) > 50 else sentence
                            })
                        
                        # Cache these generated timestamps
                        try:
                            timestamp_data = {
                                "video_id": video_id,
                                "formatted_timestamps": timestamps
                            }
                            with open(timestamp_path, "w") as f:
                                json.dump(timestamp_data, f)
                        except Exception as write_error:
                            logger.error(f"Error caching generated timestamps to file: {str(write_error)}")
                        
                        return timestamps
            except Exception as gen_error:
                logger.error(f"Error generating timestamps from transcription: {str(gen_error)}")
            
            return []
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching timestamps: {e}")
        return []

# Add direct socket test endpoint for debugging
@router.get("/test-socket/{video_id}", response_model=Dict[str, Any])
async def test_socket(
    video_id: str,
    tab_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Test socket.io communication for a specific video
    Sends test transcription data directly through Socket.IO
    """
    logger.info(f"Testing socket for video_id: {video_id}, tab_id: {tab_id}")
    
    try:
        # Try to emit messages through socket.io
        from app.socketio_server import sio
        
        # Create test data
        test_timestamps = [
            {"word": "This", "start_time": 0.0, "end_time": 0.5},
            {"word": "is", "start_time": 0.5, "end_time": 0.7},
            {"word": "a", "start_time": 0.7, "end_time": 0.8},
            {"word": "test", "start_time": 0.8, "end_time": 1.2},
            {"word": "message", "start_time": 1.2, "end_time": 2.0}
        ]
        
        test_transcript = f"This is a test message for video_id: {video_id}"
        
        # Send test messages
        if tab_id:
            await sio.emit('transcription_status', {
                'status': 'completed',
                'message': 'Test transcription completed',
                'video_id': video_id,
                'progress': 100
            }, room=tab_id)
            
            await sio.emit('transcription', {
                'status': 'success',
                'video_id': video_id,
                'transcript': test_transcript,
                'has_timestamps': True
            }, room=tab_id)
            
            await sio.emit('timestamps_available', {
                'video_id': video_id
            }, room=tab_id)
            
            logger.info(f"Sent test messages to tab_id: {tab_id}")
        else:
            # Broadcast to all clients 
            await sio.emit('transcription_status', {
                'status': 'completed',
                'message': 'Test transcription completed',
                'video_id': video_id,
                'progress': 100
            })
            
            await sio.emit('transcription', {
                'status': 'success',
                'video_id': video_id,
                'transcript': test_transcript,
                'has_timestamps': True
            })
            
            await sio.emit('timestamps_available', {
                'video_id': video_id
            })
            
            logger.info("Sent test messages broadcasted to all clients")
        
        return {
            "success": True,
            "message": "Test messages sent through socket.io",
            "video_id": video_id,
            "tab_id": tab_id
        }
        
    except Exception as e:
        logger.error(f"Error testing socket: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id,
            "tab_id": tab_id
        }

# Add direct API transcription endpoint for debugging
@router.post("/{video_id}/generate-transcription", response_model=Dict[str, Any])
async def generate_transcription(
    video_id: str,
    tab_id: Optional[str] = Form(None),
    force_debug: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Force generate transcription for a video, bypassing socket.io
    Useful for debugging transcription issues
    """
    # Verify video exists and belongs to user
    from sqlalchemy import text
    
    result = db.execute(
        text("SELECT video_url, user_id FROM videos WHERE id::text = :id"),
        {"id": str(video_id)}
    )
    
    video_row = result.fetchone()
    
    if not video_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
        
    if str(video_row[1]) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    video_path = video_row[0]
    
    # Create transcription file path
    output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
    
    # Ensure debug mode is enabled if requested
    original_debug_value = os.environ.get("DEBUG_TRANSCRIPTION", "0")
    if force_debug:
        os.environ["DEBUG_TRANSCRIPTION"] = "1"
    
    try:
        # Create mock transcription directly
        mock_timestamps = [
            {"word": "This", "start_time": 0.0, "end_time": 0.5},
            {"word": "is", "start_time": 0.5, "end_time": 0.7},
            {"word": "a", "start_time": 0.7, "end_time": 0.8},
            {"word": "direct", "start_time": 0.8, "end_time": 1.2},
            {"word": "API", "start_time": 1.2, "end_time": 1.5},
            {"word": "transcription", "start_time": 1.5, "end_time": 2.0},
            {"word": "for", "start_time": 2.0, "end_time": 2.2},
            {"word": "video", "start_time": 2.2, "end_time": 2.5},
            {"word": "ID:", "start_time": 2.5, "end_time": 2.7},
            {"word": video_id, "start_time": 2.7, "end_time": 3.2}
        ]
        
        transcription_result = {
            "text": f"This is a direct API transcription for video ID: {video_id}",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "text": f"This is a direct API transcription for video ID: {video_id}"
                }
            ],
            "timestamps": mock_timestamps,
            "video_id": video_id
        }
        
        # Save the mock transcript to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(transcription_result, f)
        
        # Save timestamps separately
        timestamps_dir = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps")
        os.makedirs(timestamps_dir, exist_ok=True)
        timestamp_file = os.path.join(timestamps_dir, f"{video_id}_timestamps.json")
        
        with open(timestamp_file, "w") as f:
            json.dump({
                "video_id": video_id,
                "formatted_timestamps": [
                    {
                        "time": ts["start_time"],
                        "time_formatted": f"{int(ts['start_time'] // 60)}:{int(ts['start_time'] % 60):02d}",
                        "text": ts["word"]
                    } for ts in mock_timestamps
                ],
                "raw_timestamps": mock_timestamps,
                "created_at": time.time()
            }, f)
        
        # Notify client of completion through socket if tab_id is provided
        if tab_id:
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': 'completed',
                    'message': 'Direct API transcription completed',
                    'video_id': video_id,
                    'progress': 100
                }, room=tab_id)
                
                await sio.emit('transcription', {
                    'status': 'success',
                    'video_id': video_id,
                    'transcript': transcription_result["text"],
                    'has_timestamps': True
                }, room=tab_id)
                
                await sio.emit('timestamps_available', {
                    'video_id': video_id
                }, room=tab_id)
                
                logger.info(f"Sent mock transcription data via direct API to client for {video_id}")
            except Exception as notify_error:
                logger.error(f"Error sending mock transcription via sockets: {str(notify_error)}")
        
        # Now try to run the real transcription process if not forcing debug
        if not force_debug:
            try:
                # Import and run the process in the background
                import asyncio
                
                asyncio.create_task(
                    process_audio_file(
                        file_path=video_path,
                        output_file=output_file,
                        language_code="en",
                        video_id=video_id,
                        tab_id=tab_id,
                        session_id=tab_id,
                        file_type="video"
                    )
                )
                
                logger.info(f"Started real transcription process in background for {video_id}")
            except Exception as process_error:
                logger.error(f"Error starting real transcription: {str(process_error)}")
        
        return {
            "success": True,
            "message": "Direct API transcription completed",
            "video_id": video_id,
            "tab_id": tab_id
        }
        
    except Exception as e:
        logger.error(f"Error in direct API transcription: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id,
            "tab_id": tab_id
        }
    finally:
        # Restore original debug value
        os.environ["DEBUG_TRANSCRIPTION"] = original_debug_value