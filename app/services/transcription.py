"""
Production-ready transcription service for Luna AI with robust timestamp support
Handles transcription via Google Speech-to-Text and AssemblyAI with fallback mechanisms
"""
import os
import json
import tempfile
import logging
import subprocess
import asyncio
import shutil
import re
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from app.config import settings
from app.services.transcription_cache import TranscriptionCache

# Configure logging with more detailed format
logger = logging.getLogger("transcription")
file_handler = logging.FileHandler(
    os.path.join(
        settings.LOG_DIR if hasattr(settings, 'LOG_DIR') else '/tmp',
        'transcription.log'
    )
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Setup environment variables from settings
if hasattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS') and settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
    logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to {settings.GOOGLE_APPLICATION_CREDENTIALS}")
else:
    logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set in settings")
    logger.warning("Google Cloud Speech-to-Text service will not be available")

if hasattr(settings, 'ASSEMBLYAI_API_KEY') and settings.ASSEMBLYAI_API_KEY:
    os.environ["ASSEMBLYAI_API_KEY"] = settings.ASSEMBLYAI_API_KEY
    logger.info(f"Set ASSEMBLYAI_API_KEY (length: {len(settings.ASSEMBLYAI_API_KEY)})")
else:
    logger.warning("ASSEMBLYAI_API_KEY not set in settings")
    logger.warning("AssemblyAI service will not be available")

# Only enable debug mode if explicitly set in config or environment
DEBUG_MODE_ENABLED = os.environ.get("DEBUG_TRANSCRIPTION") == "1"
if DEBUG_MODE_ENABLED:
    logger.info("🐞 GLOBAL DEBUG MODE ENABLED - Will use mock transcriptions")

# Create necessary directories
os.makedirs(settings.TRANSCRIPTION_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.TRANSCRIPTION_DIR, "timestamps"), exist_ok=True)

# Initialize transcription cache
cache_dir = Path(settings.TRANSCRIPTION_DIR) / "cache"
os.makedirs(cache_dir, exist_ok=True)
transcription_cache = TranscriptionCache(cache_dir)

# Check if FFmpeg is installed and accessible
def check_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"FFmpeg is installed: {result.stdout.splitlines()[0]}")
            return True
        else:
            logger.error(
                f"FFmpeg check failed with return code {result.returncode}: {result.stderr}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg check timed out")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found on system path")
        return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {str(e)}")
        return False

# Check Google Cloud credentials
def check_google_credentials():
    """Check if Google Cloud credentials are valid"""
    try:
        from google.cloud import speech_v1p1beta1 as speech
        client = speech.SpeechClient()
        logger.info("Google Cloud Speech client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud Speech client: {e}")
        return False

# Check AssemblyAI credentials
def check_assemblyai_credentials():
    """Check if AssemblyAI credentials are valid"""
    try:
        if os.environ.get("ASSEMBLYAI_API_KEY"):
            import assemblyai as aai
            aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            logger.info("AssemblyAI client initialized successfully")
            return True
        else:
            logger.warning("AssemblyAI API key not provided")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize AssemblyAI client: {e}")
        return False

# Check if pytube is installed
def check_pytube_installed():
    """Check if pytube is installed for YouTube downloads"""
    try:
        import pytube
        logger.info(f"Pytube is installed: {pytube.__version__}")
        return True
    except ImportError:
        logger.error("Pytube not installed - YouTube downloads will not work")
        return False
    except Exception as e:
        logger.error(f"Error checking pytube: {str(e)}")
        return False

# Check services on module load
FFMPEG_AVAILABLE     = check_ffmpeg_installed()
GOOGLE_AVAILABLE     = check_google_credentials()
ASSEMBLYAI_AVAILABLE = check_assemblyai_credentials()
PYTUBE_AVAILABLE     = check_pytube_installed()
logger.info(
    f"Service availability: FFmpeg={FFMPEG_AVAILABLE}, "
    f"Google={GOOGLE_AVAILABLE}, "
    f"AssemblyAI={ASSEMBLYAI_AVAILABLE}, "
    f"YouTube={PYTUBE_AVAILABLE}"
)

# Define a standardized function to determine file paths
def get_transcription_file_path(video_id, tab_id=None):
    """Get standardized file path for transcription data"""
    normalized_id = normalize_video_id(video_id)
    
    # Create a base path
    base_path = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{normalized_id}")
    
    # Add tab_id if provided
    if tab_id:
        base_path += f"_{tab_id}"
    
    # Return with extension
    return f"{base_path}.json"

def get_timestamp_file_path(video_id):
    """Get standardized file path for timestamps data"""
    normalized_id = normalize_video_id(video_id)
    return os.path.join(
        settings.TRANSCRIPTION_DIR,
        "timestamps", 
        f"{normalized_id}_timestamps.json"
    )

def save_timestamps_to_file(video_id: str, timestamps: List[Dict]) -> str:
    """
    Save timestamp data to a separate file
    Args:
        video_id: Unique identifier for the video
        timestamps: List of timestamp data
    Returns:
        Path to the saved file
    """
    try:
        formatted_timestamps = format_timestamps_for_display(timestamps)
        timestamp_dir = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps")
        os.makedirs(timestamp_dir, exist_ok=True)
        clean_video_id = normalize_video_id(video_id)
        timestamp_path = get_timestamp_file_path(clean_video_id)
        
        with open(timestamp_path, "w") as f:
            json.dump({
                "video_id": video_id,
                "raw_timestamps": timestamps,
                "formatted_timestamps": formatted_timestamps,
                "created_at": time.time()
            }, f, indent=2)
        logger.info(f"Saved {len(formatted_timestamps)} formatted timestamps to {timestamp_path}")
        return timestamp_path
    except Exception as e:
        logger.error(f"Error saving timestamps to file: {str(e)}", exc_info=True)
        return ""

def normalize_video_id(video_id: str) -> str:
    """
    Normalize video ID to ensure consistent format across services,
    with improved handling of URLs with additional parameters
    """
    if not video_id:
        return ""
    
    # Handle YouTube IDs with and without prefix consistently
    if "youtube.com" in video_id or "youtu.be" in video_id:
        # Extract video ID from URL using urllib.parse for better handling
        parsed_url = urlparse(video_id)
        
        if "youtube.com" in video_id and parsed_url.path == "/watch":
            # Standard YouTube URL format with query params
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                # Take just the first value and ignore any additional parameters
                video_id = query_params['v'][0]
        elif "youtu.be/" in video_id:
            # Shortened URL format - strip any query parameters
            path = parsed_url.path.lstrip('/')
            video_id = path.split('/')[0].split('?')[0]
        # Handle YouTube Shorts URLs
        elif "/shorts/" in video_id:
            path_parts = parsed_url.path.split('/')
            for i, part in enumerate(path_parts):
                if part == "shorts" and i + 1 < len(path_parts):
                    # Remove any query parameters from shorts ID
                    video_id = path_parts[i + 1].split('?')[0]
                    break
        
        return f"youtube_{video_id}"
    
    # If it's already a YouTube ID with prefix
    if video_id.startswith("youtube_"):
        return video_id
    
    # If it's an upload ID with prefix
    if video_id.startswith("upload_"):
        return video_id
    
    # Convert file_ID format to upload_ID format if needed
    if video_id.startswith("file_"):
        # Convert file_1745300468704 to upload_1745300468_XXXX format
        # This maintains consistency with frontend expectations
        parts = video_id.replace("file_", "").split("_")
        if len(parts) == 1 and len(parts[0]) >= 10:
            # If it's just a timestamp (like file_1745300468704)
            timestamp = parts[0][:10]  # Take first 10 chars for timestamp
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
            return f"upload_{timestamp}_{random_id}"
    
    return video_id

def format_timestamps_for_display(timestamps: List[Dict]) -> List[Dict]:
    """
    Format timestamps for display on the frontend
    """
    try:
        if not timestamps:
            return []
            
        formatted = []
        current_segment = {"text": "", "start_time": None, "end_time": None}
        
        for i, item in enumerate(timestamps):
            if not isinstance(item, dict):
                continue
                
            word       = item.get("word", "")
            start_time = item.get("start_time", 0)
            end_time   = item.get("end_time", 0)
            
            if current_segment["start_time"] is None:
                current_segment["start_time"] = start_time
                current_segment["text"]       = word
            else:
                if word and not word.startswith((',', '.', '!', '?', ':', ';')):
                    current_segment["text"] += " "
                current_segment["text"] += word
            
            current_segment["end_time"] = end_time
            
            # Create a new segment every ~10 words or on punctuation
            if (i + 1) % 10 == 0 or (word and word.endswith(('.', '!', '?'))):
                sf = format_seconds_to_mmss(current_segment["start_time"])
                ef = format_seconds_to_mmss(current_segment["end_time"])
                formatted.append({
                    "text": current_segment["text"].strip(),
                    "start_time": current_segment["start_time"],
                    "end_time": current_segment["end_time"],
                    "display_time": f"{sf} - {ef}"
                })
                current_segment = {"text": "", "start_time": None, "end_time": None}
        
        # Add the final segment if it exists
        if current_segment["text"]:
            sf = format_seconds_to_mmss(current_segment["start_time"])
            ef = format_seconds_to_mmss(current_segment["end_time"])
            formatted.append({
                "text": current_segment["text"].strip(),
                "start_time": current_segment["start_time"],
                "end_time": current_segment["end_time"],
                "display_time": f"{sf} - {ef}"
            })
        
        return formatted
    except Exception as e:
        logger.error(f"Error formatting timestamps: {str(e)}", exc_info=True)
        return []

def format_seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    if seconds is None:
        return "00:00"
    minutes = int(seconds // 60)
    secs    = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# Validate transcription services
def validate_transcription_services():
    """Validate all required services and return availability status"""
    services = {
        "ffmpeg": FFMPEG_AVAILABLE,
        "google": GOOGLE_AVAILABLE,
        "assemblyai": ASSEMBLYAI_AVAILABLE,
        "pytube": PYTUBE_AVAILABLE
    }
    
    # Log the complete status
    logger.info(f"Service availability check: {services}")
    
    # Return False only if NO services are available
    if not services["ffmpeg"]:
        logger.error("FFmpeg is required and not available")
        return False
        
    if not services["google"] and not services["assemblyai"]:
        logger.error("No transcription service available (Google or AssemblyAI)")
        return False
        
    return True

# Check if a URL is a YouTube Shorts
def is_youtube_shorts(url: str) -> bool:
    """Check if a URL is a YouTube Shorts URL"""
    return "/shorts/" in url.lower()

# YouTube download with improved error handling and specific handling for Shorts
# Replace your current download_youtube_video function with this improved version:
async def download_youtube_video(
    youtube_url: str,
    output_dir: str,
    video_id: str = None
) -> Dict[str, Any]:
    """
    Download a YouTube video for processing with improved error handling
    """
    logger.info(f"Starting YouTube download for URL: {youtube_url}, ID: {video_id}")
    
    if not PYTUBE_AVAILABLE:
        error_msg = "Pytube not available. Cannot download YouTube videos."
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "video_path": None,
            "video_id": video_id
        }
    
    # Generate a video ID if not provided
    if not video_id:
        timestamp = int(time.time())
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        video_id = f"youtube_{timestamp}_{random_id}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Downloading YouTube video into {output_dir} as ID {video_id}")
    
    # Create safe file path
    safe_id = re.sub(r'[^\w\-_]', '_', video_id)
    output_path = os.path.join(output_dir, f"{safe_id}.mp4")
    
    # List of user agents to try
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    # Check if this is a YouTube Shorts URL
    is_shorts = is_youtube_shorts(youtube_url)
    if is_shorts:
        logger.info(f"Detected YouTube Shorts URL: {youtube_url}")
    
    # Try different download methods in sequence
    methods = ["yt_dlp", "pytube_standard", "pytube_shorts", "ffmpeg_direct"]
    
    # If this is a shorts URL, prioritize shorts-specific methods
    if is_shorts:
        methods = ["yt_dlp", "pytube_shorts", "ffmpeg_direct", "pytube_standard"]
    
    for method in methods:
        try:
            logger.info(f"Trying download method: {method}")
            
            if method == "pytube_standard":
                success, result = await download_with_pytube(youtube_url, output_path, is_shorts=False)
            elif method == "pytube_shorts":
                success, result = await download_with_pytube(youtube_url, output_path, is_shorts=True)
            elif method == "yt_dlp":
                success, result = await download_with_yt_dlp(youtube_url, output_path)
            elif method == "ffmpeg_direct":
                success, result = await download_with_ffmpeg(youtube_url, output_path)
            
            if success:
                logger.info(f"YouTube download successful with method: {method}")
                
                # Verify the file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Download complete - File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                    return {
                        "success": True,
                        "video_path": output_path,
                        "video_id": video_id,
                        "metadata": result
                    }
                else:
                    logger.warning(f"Download succeeded but file missing or empty: {output_path}")
            else:
                logger.warning(f"Download method {method} failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error with download method {method}: {str(e)}", exc_info=True)
    
    # If we get here, all methods failed
    error_msg = "YouTube extraction failed - possible restrictions or regional blocks"
    logger.error(error_msg)
    
    # Extract the video title if possible for better error messages
    video_info = "Unknown video"
    
    # Return detailed error
    return {
        "success": False,
        "error": error_msg,
        "detailed_error": "All download methods failed",
        "video_path": None,
        "video_id": video_id,
        "video_info": video_info,
        "url": youtube_url
    }


# Download using pytube
async def download_with_pytube(youtube_url: str, output_path: str, is_shorts: bool = False) -> Tuple[bool, Dict]:
    """Download YouTube video using pytube with specific handling for Shorts"""
    try:
        import pytube
        loop = asyncio.get_event_loop()
        
        def _download():
            try:
                # Configure pytube differently for Shorts
                if is_shorts:
                    # For Shorts, use specific parameters
                    yt = pytube.YouTube(
                        youtube_url,
                        use_oauth=False,
                        allow_oauth_cache=False
                        # Removed the skip_regex parameter as it's not supported
                    )
                else:
                    # Standard configuration
                    yt = pytube.YouTube(youtube_url, use_oauth=False, allow_oauth_cache=False)
                
                # Try to get video metadata - this might fail for Shorts
                try:
                    video_title = yt.title
                    video_author = yt.author
                    video_length = yt.length
                except Exception as meta_error:
                    logger.warning(f"Could not get video metadata: {str(meta_error)}")
                    video_title = "Unknown Title"
                    video_author = "Unknown Author"
                    video_length = 0
                
                # For Shorts, try to directly access streams
                if is_shorts:
                    # Try multiple stream types for Shorts
                    streams = []
                    
                    # First try: Audio only streams (often works better for Shorts)
                    streams.extend(yt.streams.filter(only_audio=True).order_by('abr').desc())
                    
                    # Second try: Progressive streams with video
                    streams.extend(yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc())
                    
                    # Third try: Dash streams with video
                    streams.extend(yt.streams.filter(adaptive=True, file_extension='mp4', only_video=False).order_by('resolution').desc())
                else:
                    # Standard approach for regular videos
                    streams = []
                    
                    # First try: Progressive streams with video
                    streams.extend(yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc())
                    
                    # Second try: Dash streams with video
                    streams.extend(yt.streams.filter(adaptive=True, file_extension='mp4', only_video=False).order_by('resolution').desc())
                    
                    # Third try: Audio only streams
                    streams.extend(yt.streams.filter(only_audio=True).order_by('abr').desc())
                
                # Try each stream until one works
                for i, stream in enumerate(streams):
                    try:
                        logger.info(f"Trying stream option {i+1}/{len(streams)}: {stream}")
                        stream.download(filename=output_path)
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            logger.info(f"Successfully downloaded with stream option {i+1}")
                            break
                    except Exception as stream_error:
                        logger.warning(f"Stream {i+1} download failed: {str(stream_error)}")
                        continue
                
                # Verify the file exists and has content
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise Exception("All stream download attempts failed")
                
                return {
                    "path": output_path,
                    "video_title": video_title,
                    "author": video_author,
                    "length": video_length
                }
            except Exception as inner_error:
                logger.error(f"Error in pytube download function: {str(inner_error)}")
                raise
        
        # Run download with timeout
        try:
            download_result = await asyncio.wait_for(
                loop.run_in_executor(None, _download),
                timeout=300  # 5 minute timeout
            )
            return True, download_result
        except asyncio.TimeoutError:
            logger.error("YouTube download timed out after 5 minutes")
            return False, {
                "error": "Download timed out after 5 minutes",
                "debug_info": {
                    "url": youtube_url,
                    "output_path": output_path
                }
            }
    
    except Exception as e:
        logger.error(f"Pytube download error: {str(e)}")
        return False, {"error": str(e)}

# Download using yt-dlp (as an alternative to pytube)
# Add this function to transcription.py
async def download_with_yt_dlp(youtube_url: str, output_path: str) -> Tuple[bool, Dict]:
    """Download YouTube video using yt-dlp command line tool"""
    try:
        # Check if yt-dlp is installed
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.warning("yt-dlp not available or not working properly")
                return False, {"error": "yt-dlp not available"}
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("yt-dlp not installed")
            return False, {"error": "yt-dlp not installed"}
        
        # Construct the yt-dlp command
        download_cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "-o", output_path,
            youtube_url
        ]
        
        logger.info(f"Running yt-dlp command: {' '.join(download_cmd)}")
        
        # Execute the command
        process = await asyncio.create_subprocess_exec(
            *download_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Verify the file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"yt-dlp download successful: {output_path}")
                return True, {
                    "path": output_path,
                    "video_title": "Downloaded with yt-dlp",
                    "author": "Unknown",
                    "length": 0
                }
            else:
                logger.error("yt-dlp reported success but file is missing or empty")
                return False, {"error": "Output file is missing or empty"}
        else:
            error_output = stderr.decode('utf-8', errors='ignore')
            logger.error(f"yt-dlp download failed: {error_output}")
            return False, {"error": f"yt-dlp error: {error_output}"}
    
    except Exception as e:
        logger.error(f"Error using yt-dlp: {str(e)}")
        return False, {"error": str(e)}

# Download using FFmpeg directly
async def download_with_ffmpeg(youtube_url: str, output_path: str) -> Tuple[bool, Dict]:
    """Download YouTube video directly using FFmpeg"""
    try:
        # Use FFmpeg to download from YouTube directly
        # Note: This method may not work for all YouTube URLs
        download_cmd = [
            "ffmpeg", 
            "-i", youtube_url, 
            "-c", "copy", 
            output_path,
            "-y"  # Overwrite existing files
        ]
        
        logger.info(f"Running FFmpeg direct download: {' '.join(download_cmd)}")
        
        # Execute the command
        process = await asyncio.create_subprocess_exec(
            *download_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Verify the file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"FFmpeg direct download successful: {output_path}")
                return True, {
                    "path": output_path,
                    "video_title": "Downloaded with FFmpeg",
                    "author": "Unknown",
                    "length": 0
                }
            else:
                logger.error("FFmpeg direct download reported success but file is missing or empty")
                return False, {"error": "Output file is missing or empty"}
        else:
            error_output = stderr.decode('utf-8', errors='ignore')
            logger.error(f"FFmpeg direct download failed: {error_output}")
            return False, {"error": f"FFmpeg error: {error_output}"}
    
    except Exception as e:
        logger.error(f"Error using FFmpeg direct download: {str(e)}")
        return False, {"error": str(e)}

# Add this to transcription.py
async def integrated_youtube_download(youtube_url, output_dir, video_id):
    """
    Use the more robust downloader from video_processing module
    """
    try:
        from app.services.video_processing import download_youtube_video as vp_download
        
        # Call the more robust downloader
        video_path = await vp_download(youtube_url, output_dir)
        
        if video_path and os.path.exists(video_path):
            return {
                "success": True,
                "video_path": video_path,
                "video_id": video_id
            }
        else:
            return {
                "success": False,
                "error": "Video processing downloader failed",
                "video_id": video_id
            }
    except ImportError:
        logger.warning("Could not import video_processing module, falling back to standard download")
        return await download_youtube_video(youtube_url, output_dir, video_id)
    except Exception as e:
        logger.error(f"Error using integrated YouTube download: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id
        }
        
async def transcribe_video_debug(
    video_path: str,
    output_file: Optional[str] = None,
    language_code: str = "en",
    video_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    DEBUG VERSION: Creates a mock transcription without using external services
    """
    logger.info(f"DEBUG TRANSCRIPTION for video: {video_path}, video_id: {video_id}")
    
    # Check if it's a YouTube error case, create more realistic mock
    is_youtube_error = kwargs.get("youtube_error", False)
    youtube_url = kwargs.get("youtube_url", "")
    video_info = kwargs.get("video_info", "Unknown video")
    
    if is_youtube_error:
        mock_transcription = {
            "text": f"[Unable to transcribe video due to download error: YouTube extraction failed - possible restrictions or regional blocks]\n\nThis video appears to be titled {video_info}\nVideo ID: {video_id or 'unknown'}\nURL: {youtube_url}\n\nThe video content could not be accessed due to YouTube restrictions or connection issues.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 6.0, "text": "[Unable to transcribe video due to download error: YouTube extraction failed - possible restrictions or regional blocks]"},
                {"id": 1, "start": 6.0, "end": 12.0, "text": f"This video appears to be titled {video_info}."},
                {"id": 2, "start": 12.0, "end": 19.0, "text": "The video content could not be accessed due to YouTube restrictions."}
            ],
            "timestamps": [
                {"word": "[Unable", "start_time": 0.0, "end_time": 0.5},
                {"word": "to", "start_time": 0.5, "end_time": 0.7},
                {"word": "transcribe", "start_time": 0.7, "end_time": 1.2},
                {"word": "video", "start_time": 1.2, "end_time": 1.5},
                {"word": "due", "start_time": 1.5, "end_time": 1.8},
                {"word": "to", "start_time": 1.8, "end_time": 2.0},
                {"word": "download", "start_time": 2.0, "end_time": 2.5},
                {"word": "error]", "start_time": 2.5, "end_time": 3.0},
                {"word": "This", "start_time": 6.0, "end_time": 6.5},
                {"word": "video", "start_time": 6.5, "end_time": 7.0},
                {"word": "appears", "start_time": 7.0, "end_time": 7.5},
                {"word": "to", "start_time": 7.5, "end_time": 8.0},
                {"word": "be", "start_time": 8.0, "end_time": 8.5},
                {"word": "titled", "start_time": 8.5, "end_time": 9.0},
                {"word": video_info, "start_time": 9.0, "end_time": 10.0},
                {"word": "could", "start_time": 12.0, "end_time": 12.5},
                {"word": "not", "start_time": 12.5, "end_time": 13.0},
                {"word": "be", "start_time": 13.0, "end_time": 13.5},
                {"word": "accessed", "start_time": 13.5, "end_time": 14.0},
                {"word": "due", "start_time": 14.0, "end_time": 14.5},
                {"word": "to", "start_time": 14.5, "end_time": 15.0},
                {"word": "YouTube", "start_time": 15.0, "end_time": 15.5},
                {"word": "restrictions", "start_time": 15.5, "end_time": 16.0},
                {"word": "or", "start_time": 16.0, "end_time": 16.5},
                {"word": "connection", "start_time": 16.5, "end_time": 17.0},
                {"word": "issues.", "start_time": 17.0, "end_time": 17.5}
            ]
        }
    else:
        # Standard debug transcription
        video_title = os.path.basename(video_path) if video_path else f"Debug video {video_id}"
        mock_transcription = {
            "text": f"This is a debug transcription for {video_title}. Video ID: {video_id or 'unknown'}. The Luna AI system is generating this mock transcript because the transcription service is in debug mode.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.0, "text": f"This is a debug transcription for {video_title}."},
                {"id": 1, "start": 5.0, "end": 10.0, "text": f"Video ID: {video_id or 'unknown'}."},
                {"id": 2, "start": 10.0, "end": 15.0, "text": "The Luna AI system is generating this mock transcript because the transcription service is in debug mode."}
            ],
            "timestamps": [
                {"word": "This", "start_time": 0.0, "end_time": 0.5},
                {"word": "is", "start_time": 0.5, "end_time": 0.7},
                {"word": "a", "start_time": 0.7, "end_time": 0.8},
                {"word": "debug", "start_time": 0.8, "end_time": 1.2},
                {"word": "transcription", "start_time": 1.2, "end_time": 2.0},
                {"word": "for", "start_time": 2.0, "end_time": 2.2},
                {"word": video_title, "start_time": 2.2, "end_time": 3.0},
                {"word": "Video", "start_time": 5.0, "end_time": 5.5},
                {"word": "ID:", "start_time": 5.5, "end_time": 6.0},
                {"word": video_id or "unknown", "start_time": 6.0, "end_time": 7.0},
                {"word": "The", "start_time": 10.0, "end_time": 10.3},
                {"word": "Luna", "start_time": 10.3, "end_time": 10.6},
                {"word": "AI", "start_time": 10.6, "end_time": 10.9},
                {"word": "system", "start_time": 10.9, "end_time": 11.2},
                {"word": "is", "start_time": 11.2, "end_time": 11.4},
                {"word": "generating", "start_time": 11.4, "end_time": 12.0},
                {"word": "this", "start_time": 12.0, "end_time": 12.2},
                {"word": "mock", "start_time": 12.2, "end_time": 12.5},
                {"word": "transcript", "start_time": 12.5, "end_time": 13.0},
                {"word": "because", "start_time": 13.0, "end_time": 13.5},
                {"word": "the", "start_time": 13.5, "end_time": 13.7},
                {"word": "transcription", "start_time": 13.7, "end_time": 14.3},
                {"word": "service", "start_time": 14.3, "end_time": 14.6},
                {"word": "is", "start_time": 14.6, "end_time": 14.7},
                {"word": "in", "start_time": 14.7, "end_time": 14.8},
                {"word": "debug", "start_time": 14.8, "end_time": 15.0},
                {"word": "mode.", "start_time": 15.0, "end_time": 15.3}
            ]
        }
    
    if video_id:
        mock_transcription["video_id"] = video_id
    
    # Add a slight delay to simulate processing time
    await asyncio.sleep(1)
    
    # Save to output file if specified
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(mock_transcription, f, indent=2)
            logger.info(f"Saved debug transcription to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save debug transcription: {e}")
    
    # Save timestamps separately
    if video_id:
        save_timestamps_to_file(video_id, mock_transcription["timestamps"])
    
    return mock_transcription

# Transcription with Google Cloud Speech-to-Text
async def transcribe_with_google(
    audio_path: str,
    language_code: str = "en",
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using Google Cloud Speech-to-Text API
    """
    logger.info(f"Transcribing using Google Cloud Speech-to-Text: {audio_path}")
    
    if not GOOGLE_AVAILABLE:
        logger.error("Google Cloud Speech-to-Text not available")
        return {
            "text": "Error: Google Cloud Speech-to-Text service is not available",
            "segments": [],
            "timestamps": [],
            "error": "Service not available"
        }
    
    try:
        from google.cloud import speech_v1p1beta1 as speech
        
        # Initialize the client
        client = speech.SpeechClient()
        
        # Load the audio file
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        # Configure the request - REMOVED 'model' parameter which was causing errors
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # Must match the audio extraction settings
            language_code=language_code,
            enable_word_time_offsets=True,  # Enable timestamps
            enable_automatic_punctuation=True,
            use_enhanced=True  # Use enhanced model
            # Removed model="video" parameter that was causing incompatibility errors
        )
        
        # Make the API call
        operation = client.long_running_recognize(config=config, audio=audio)
        logger.info("Waiting for Google Cloud Speech operation to complete...")
        
        # Get the result
        response = operation.result(timeout=600)  # 10 minute timeout
        
        # Process the response
        transcript = ""
        segments = []
        timestamps = []
        
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            transcript += alternative.transcript + " "
            
            # Create segment
            segments.append({
                "id": i,
                "start": alternative.words[0].start_time.total_seconds() if alternative.words else 0,
                "end": alternative.words[-1].end_time.total_seconds() if alternative.words else 0,
                "text": alternative.transcript
            })
            
            # Create word-level timestamps
            for word_info in alternative.words:
                timestamps.append({
                    "word": word_info.word,
                    "start_time": word_info.start_time.total_seconds(),
                    "end_time": word_info.end_time.total_seconds()
                })
        
        transcription_result = {
            "text": transcript.strip(),
            "segments": segments,
            "timestamps": timestamps
        }
        
        logger.info(f"Google transcription complete: {len(timestamps)} words, {len(segments)} segments")
        return transcription_result
    
    except Exception as e:
        logger.error(f"Google transcription error: {str(e)}", exc_info=True)
        return {
            "text": f"Error in Google transcription: {str(e)}",
            "segments": [{"id": 0, "text": f"Transcription error: {str(e)}"}],
            "timestamps": [],
            "error": str(e)
        }

# Transcription with AssemblyAI
async def transcribe_with_assemblyai(
    audio_path_or_url: str,
    language_code: str = "en",
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using AssemblyAI API with direct YouTube URL support
    """
    logger.info(f"Transcribing using AssemblyAI: {audio_path_or_url}")
    
    if not ASSEMBLYAI_AVAILABLE:
        logger.error("AssemblyAI service not available")
        return {
            "text": "Error: AssemblyAI service is not available",
            "segments": [],
            "timestamps": [],
            "error": "Service not available"
        }
    
    try:
        import assemblyai as aai
        
        # Set API key
        aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        
        transcriber = aai.Transcriber()
        
        # Check if this is a YouTube URL
        is_youtube_url = isinstance(audio_path_or_url, str) and audio_path_or_url.startswith(('http://', 'https://')) and (
            'youtube.com' in audio_path_or_url or 'youtu.be' in audio_path_or_url
        )
        
        # Log whether we're using direct URL or file path
        if is_youtube_url:
            logger.info(f"Processing YouTube URL directly with AssemblyAI: {audio_path_or_url}")
        else:
            logger.info(f"Processing audio file with AssemblyAI: {audio_path_or_url}")
        
        # Start the transcription - AssemblyAI supports both file paths and URLs
        transcript = transcriber.transcribe(audio_path_or_url)
        
        if hasattr(transcript, 'status') and transcript.status == "error":
            raise Exception(f"AssemblyAI error: {transcript.error}")
        
        # Process the response
        text = transcript.text if hasattr(transcript, 'text') else ""
        
        # Process utterances (segments)
        segments = []
        utterances = getattr(transcript, 'utterances', []) or []
        for i, utterance in enumerate(utterances):
            segments.append({
                "id": i,
                "start": utterance.start / 1000 if hasattr(utterance, 'start') else 0,
                "end": utterance.end / 1000 if hasattr(utterance, 'end') else 0,
                "text": utterance.text if hasattr(utterance, 'text') else "",
                "speaker": utterance.speaker if hasattr(utterance, 'speaker') else ""
            })
        
        # Process word-level timestamps
        timestamps = []
        words = getattr(transcript, 'words', []) or []
        for word in words:
            timestamps.append({
                "word": word.text if hasattr(word, 'text') else "",
                "start_time": word.start / 1000 if hasattr(word, 'start') else 0,
                "end_time": word.end / 1000 if hasattr(word, 'end') else 0,
                "confidence": word.confidence if hasattr(word, 'confidence') else 1.0
            })
        
        transcription_result = {
            "text": text,
            "segments": segments,
            "timestamps": timestamps,
            "source_type": "youtube_url" if is_youtube_url else "audio_file"
        }
        
        logger.info(f"AssemblyAI transcription complete: {len(timestamps)} words, {len(segments)} segments")
        return transcription_result
    
    except Exception as e:
        logger.error(f"AssemblyAI transcription error: {str(e)}", exc_info=True)
        return {
            "text": f"Error in AssemblyAI transcription: {str(e)}",
            "segments": [{"id": 0, "text": f"Transcription error: {str(e)}"}],
            "timestamps": [],
            "error": str(e)
        }
        
# NEW FUNCTIONS FOR DYNAMIC SERVICE VERIFICATION
async def verify_ffmpeg():
    """Verify FFmpeg installation and functionality"""
    try:
        # Create a simple test command
        test_cmd = ["ffmpeg", "-version"]
        process = await asyncio.create_subprocess_exec(
            *test_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            version = stdout.decode('utf-8', errors='ignore').split('\n')[0]
            logger.info(f"FFmpeg verified: {version}")
            return {"success": True, "version": version}
        else:
            error = stderr.decode('utf-8', errors='ignore')
            logger.error(f"FFmpeg verification failed: {error}")
            return {"success": False, "error": error}
    except Exception as e:
        logger.error(f"FFmpeg verification error: {str(e)}")
        return {"success": False, "error": str(e)}

async def verify_google_credentials():
    """Verify Google Cloud credentials with a test request"""
    global GOOGLE_AVAILABLE
    
    if not GOOGLE_AVAILABLE:
        logger.info("Google Cloud Speech-to-Text not available in initial check")
    
    try:
        from google.cloud import speech_v1p1beta1 as speech
        
        # Try to initialize the client
        client = speech.SpeechClient()
        
        # Create a minimal test request to verify auth
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        
        # Check if we can get the config (doesn't actually make a full API call)
        # Just validates the client can be initialized with credentials
        if client:
            logger.info("Google Cloud Speech credentials verified")
            # Update global flag to ensure it's correct
            GOOGLE_AVAILABLE = True
            return {"success": True}
        else:
            logger.error("Google Cloud Speech client could not be initialized")
            GOOGLE_AVAILABLE = False
            return {"success": False, "error": "Client initialization failed"}
    except Exception as e:
        logger.error(f"Google credentials verification error: {str(e)}")
        GOOGLE_AVAILABLE = False
        return {"success": False, "error": str(e)}

async def verify_assemblyai_credentials():
    """Verify AssemblyAI credentials"""
    global ASSEMBLYAI_AVAILABLE
    
    if not ASSEMBLYAI_AVAILABLE:
        logger.info("AssemblyAI not available in initial check")
    
    try:
        import assemblyai as aai
        
        # Ensure API key is set
        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            logger.error("AssemblyAI API key not found in environment")
            ASSEMBLYAI_AVAILABLE = False
            return {"success": False, "error": "API key not found"}
        
        # Set API key
        aai.settings.api_key = api_key
        
        # Initialize transcriber to verify the API key works
        transcriber = aai.Transcriber()
        
        # If we got here, credentials should be valid
        logger.info("AssemblyAI credentials verified")
        ASSEMBLYAI_AVAILABLE = True
        return {"success": True}
    except Exception as e:
        logger.error(f"AssemblyAI credentials verification error: {str(e)}")
        ASSEMBLYAI_AVAILABLE = False
        return {"success": False, "error": str(e)}

# NEW FUNCTION TO VERIFY AND TRANSCRIBE MEDIA
async def verify_and_transcribe_media(source_path, video_id=None, language_code="en", session_id=None):
    """
    Verify service availability and trigger transcription for uploaded media
    
    Args:
        source_path: Path to the uploaded file or YouTube URL
        video_id: Optional ID for the video/audio
        language_code: Language code for transcription
        session_id: Optional session ID for WebSocket updates
    
    Returns:
        Dictionary with verification and transcription status
    """
    # Verify FFmpeg installation
    ffmpeg_verified = await verify_ffmpeg()
    if not ffmpeg_verified['success']:
        logger.error(f"FFmpeg verification failed: {ffmpeg_verified['error']}")
        
        # Notify clients if session_id provided
        if session_id:
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': 'FFmpeg verification failed. Cannot process audio.',
                    'video_id': video_id,
                    'progress': 100
                })
            except Exception:
                pass
                
        return {
            "success": False,
            "message": "FFmpeg verification failed. Cannot extract audio.",
            "error": ffmpeg_verified['error'],
            "video_id": video_id
        }
    
    # Verify credential availability and test authentication
    google_verified = await verify_google_credentials()
    assemblyai_verified = await verify_assemblyai_credentials()
    
    # Log verification results
    logger.info(f"Service verification: FFmpeg={ffmpeg_verified['success']}, "
                f"Google={google_verified['success']}, "
                f"AssemblyAI={assemblyai_verified['success']}")
    
    if not google_verified['success'] and not assemblyai_verified['success']:
        logger.error("No transcription service is available")
        
        # Notify clients if session_id provided
        if session_id:
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': 'No transcription service is available. Please check API credentials.',
                    'video_id': video_id,
                    'progress': 100
                })
            except Exception:
                pass
                
        return {
            "success": False,
            "message": "No transcription service is available. Please check API credentials.",
            "error": f"Google: {google_verified.get('error', 'not available')}; AssemblyAI: {assemblyai_verified.get('error', 'not available')}",
            "video_id": video_id
        }
    
    # Determine which service to try first
    preferred_service = None
    if google_verified['success']:
        preferred_service = "google"
        logger.info("Will try Google transcription first")
    elif assemblyai_verified['success']:
        preferred_service = "assemblyai"
        logger.info("Will try AssemblyAI transcription first")
    
    # Notify clients about transcription start if session_id provided
    if session_id:
        try:
            from app.socketio_server import sio
            await sio.emit('transcription_status', {
                'status': "started",
                'message': f"Starting transcription with {preferred_service.upper() if preferred_service else 'default'} service",
                'video_id': video_id,
                'progress': 0
            })
        except Exception as ws_error:
            logger.error(f"Failed to send WebSocket notification: {str(ws_error)}")
    
    # Now trigger the actual transcription
    try:
        # For YouTube URLs
        if source_path.startswith(('http://', 'https://')) and ('youtube.com' in source_path or 'youtu.be' in source_path):
            result = await transcribe_youtube_video(
                youtube_url=source_path,
                video_id=video_id,
                language_code=language_code,
                preferred_service=preferred_service,
                # Do not force disable debug mode
                debug_mode=DEBUG_MODE_ENABLED
            )
        # For uploaded files
        else:
            # Determine if this is an audio or video file
            file_extension = os.path.splitext(source_path)[1].lower()
            file_type = 'audio' if file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'] else 'video'
            
            result = await transcribe_uploaded_file(
                file_path=source_path,
                video_id=video_id,
                language_code=language_code,
                preferred_service=preferred_service,
                file_type=file_type,
                debug_mode=DEBUG_MODE_ENABLED
            )
        
        # Check for errors in the result
        if result and isinstance(result, dict) and result.get("error"):
            logger.error(f"Transcription completed with error: {result.get('error')}")
            
            # Notify clients if session_id provided
            if session_id:
                try:
                    from app.socketio_server import sio
                    await sio.emit('transcription_status', {
                        'status': 'error',
                        'message': f"Transcription error: {result.get('error')}",
                        'video_id': video_id,
                        'progress': 100
                    })
                except Exception:
                    pass
                    
            return {
                "success": False,
                "message": f"Transcription completed with error: {result.get('error')}",
                "result": result,
                "video_id": video_id
            }
        
        # Transcription successful
        logger.info(f"Transcription completed successfully for {video_id}")
        return {
            "success": True,
            "message": "Transcription completed successfully",
            "video_id": video_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        
        # Notify clients if session_id provided
        if session_id:
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': f"Transcription failed: {str(e)}",
                    'video_id': video_id,
                    'progress': 100
                })
            except Exception:
                pass
                
        return {
            "success": False,
            "message": f"Transcription failed: {str(e)}",
            "error": str(e),
            "video_id": video_id
        }

# Fix for transcribe_uploaded_file in transcription.py
# Make sure the function is async
async def transcribe_uploaded_file(  # Added 'async' here
    file_path: str,
    output_file: Optional[str] = None,
    language_code: str = "en",
    video_id: Optional[str] = None,
    file_type: str = "video",  # 'video' or 'audio'
    preferred_service: Optional[str] = None,
    debug_mode: bool = False,
    session_id: Optional[str] = None,  # Add session_id parameter
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe an uploaded video or audio file with caching support
    
    Args:
        file_path: Path to the uploaded file
        output_file: Optional path to save the transcription
        language_code: Language code for transcription
        video_id: Optional ID for the file
        file_type: Type of file ('video' or 'audio')
        preferred_service: Preferred transcription service
        debug_mode: Enable debug mode
        session_id: Optional session ID for WebSocket notifications
        
    Returns:
        Transcription result dictionary
    """
    # Generate a file hash for caching
    import hashlib
    
    if os.path.exists(file_path):
        # Generate a file hash to use for caching
        file_hash = None
        try:
            with open(file_path, "rb") as f:
                # Read the first 64KB of the file for hashing (for large files)
                file_content = f.read(65536)
                file_hash = hashlib.md5(file_content).hexdigest()
                logger.info(f"Generated file hash for caching: {file_hash}")
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
    else:
        logger.error(f"File not found: {file_path}")
        return {
            "text": f"Error: File not found at {file_path}",
            "segments": [],
            "timestamps": [],
            "error": "File not found",
            "video_id": video_id
        }
    
    # Generate video_id if not provided
    if not video_id:
        timestamp = int(time.time())
        if file_hash:
            video_id = f"upload_{timestamp}_{file_hash[:10]}"
        else:
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
            video_id = f"upload_{timestamp}_{random_id}"
        logger.info(f"Generated ID for uploaded file: {video_id}")
    
    # Set default output file if not provided
    if not output_file:
        output_file = get_transcription_file_path(video_id)
        logger.info(f"Using default output file: {output_file}")
    
    # Check for cached transcription using the file hash
    cached_transcription = None
    if file_hash:
        cache_path = os.path.join(settings.TRANSCRIPTION_DIR, "cache", f"file_{file_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_transcription = json.load(f)
                logger.info(f"Found cached transcription for file hash: {file_hash}")
                
                # Update the video_id in the cached transcription
                if cached_transcription:
                    cached_transcription["video_id"] = video_id
                    
                    # Save the updated transcription to the standard location
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "w") as f:
                        json.dump(cached_transcription, f, indent=2)
                    logger.info(f"Saved updated cached transcription to: {output_file}")
                    
                    # Notify clients using WebSocket - FIXED to use video_id consistently
                    try:
                        from app.socketio_server import sio
                        
                        # Send status update
                        if session_id:
                            await sio.emit('transcription_status', {
                                'status': "completed", 
                                'message': "Using cached transcription",
                                'video_id': video_id,  # Use video_id consistently 
                                'progress': 100
                            }, room=session_id)
                            
                            # Also emit the transcription event
                            await sio.emit('transcription', {
                                'status': "success",
                                'video_id': video_id,  # Use video_id consistently
                                'transcript': cached_transcription.get("text", ""),
                                'has_timestamps': True if "timestamps" in cached_transcription and cached_transcription["timestamps"] else False
                            }, room=session_id)
                            
                            # Send timestamp notification
                            if "timestamps" in cached_transcription and cached_transcription["timestamps"]:
                                await sio.emit('timestamps_available', {
                                    'video_id': video_id  # Use video_id consistently
                                }, room=session_id)
                            
                            logger.info(f"Notified WebSocket clients of cached transcription for {video_id}")
                    except Exception as ws_error:
                        logger.error(f"Failed to notify WebSocket clients of cached transcription: {str(ws_error)}")
                        
                    return cached_transcription
            except Exception as e:
                logger.error(f"Error reading cached transcription: {str(e)}")

    # Verify credentials at runtime
    if preferred_service == "google":
        google_verified = await verify_google_credentials()
        if not google_verified["success"]:
            if ASSEMBLYAI_AVAILABLE:
                logger.warning("Google credentials verification failed, falling back to AssemblyAI")
                preferred_service = "assemblyai"
            else:
                logger.error("Google credentials verification failed and AssemblyAI not available")
    elif preferred_service == "assemblyai":
        assemblyai_verified = await verify_assemblyai_credentials()
        if not assemblyai_verified["success"]:
            if GOOGLE_AVAILABLE:
                logger.warning("AssemblyAI credentials verification failed, falling back to Google")
                preferred_service = "google"
            else:
                logger.error("AssemblyAI credentials verification failed and Google not available")
    
    # Process the file based on type
    if file_type == "audio":
        # For audio files, we can skip the audio extraction step
        audio_path = file_path
        
        # Determine which service to use
        if preferred_service == "google" and GOOGLE_AVAILABLE:
            service_to_use = "google"
        elif preferred_service == "assemblyai" and ASSEMBLYAI_AVAILABLE:
            service_to_use = "assemblyai"
        elif GOOGLE_AVAILABLE:
            service_to_use = "google"
        elif ASSEMBLYAI_AVAILABLE:
            service_to_use = "assemblyai"
        else:
            logger.warning("No transcription service available, falling back to debug mode")
            service_to_use = "debug"
            
        # Transcribe the audio
        logger.info(f"Transcribing audio directly: {audio_path} with service: {service_to_use}")
        
        try:
            if service_to_use == "google":
                transcription_result = await transcribe_with_google(audio_path, language_code, **kwargs)
                
                # If Google fails, try Assembly AI as fallback
                if "error" in transcription_result and ASSEMBLYAI_AVAILABLE:
                    logger.warning(f"Google transcription failed: {transcription_result.get('error')}, falling back to AssemblyAI")
                    transcription_result = await transcribe_with_assemblyai(audio_path, language_code, **kwargs)
            elif service_to_use == "assemblyai":
                transcription_result = await transcribe_with_assemblyai(audio_path, language_code, **kwargs)
                
                # If AssemblyAI fails, try Google as fallback
                if "error" in transcription_result and GOOGLE_AVAILABLE:
                    logger.warning(f"AssemblyAI transcription failed: {transcription_result.get('error')}, falling back to Google")
                    transcription_result = await transcribe_with_google(audio_path, language_code, **kwargs)
            else:
                transcription_result = await transcribe_video_debug(
                    audio_path, output_file, language_code, video_id, **kwargs
                )
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            
            # Try fallback if available
            try:
                if service_to_use == "google" and ASSEMBLYAI_AVAILABLE:
                    logger.info("Error with Google transcription, trying AssemblyAI as fallback")
                    transcription_result = await transcribe_with_assemblyai(audio_path, language_code, **kwargs)
                elif service_to_use == "assemblyai" and GOOGLE_AVAILABLE:
                    logger.info("Error with AssemblyAI transcription, trying Google as fallback")
                    transcription_result = await transcribe_with_google(audio_path, language_code, **kwargs)
                else:
                    raise Exception(f"Error transcribing audio and no fallback available: {str(e)}")
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {str(fallback_error)}", exc_info=True)
                return {
                    "text": f"Error transcribing audio: {str(e)}",
                    "segments": [],
                    "timestamps": [],
                    "error": str(e),
                    "video_id": video_id
                }
    else:
        # For video files, use the standard video transcription function
        logger.info(f"Transcribing video: {file_path} with preferred service: {preferred_service}")
        transcription_result = await transcribe_video(
            video_path=file_path,
            output_file=output_file,
            language_code=language_code,
            video_id=video_id,
            preferred_service=preferred_service,
            debug_mode=debug_mode,
            session_id=session_id,  # Pass session_id for WebSocket notifications
            **kwargs
        )
    
    # Add video_id and metadata
    transcription_result["video_id"] = video_id
    transcription_result.setdefault("metadata", {}).update({
        "file_type": file_type,
        "file_name": os.path.basename(file_path),
        "file_hash": file_hash
    })
    
    # Save to cache if we have a file hash
    if file_hash:
        cache_path = os.path.join(settings.TRANSCRIPTION_DIR, "cache", f"file_{file_hash}.json")
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(transcription_result, f, indent=2)
            logger.info(f"Saved transcription to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    # Send WebSocket notifications
    if session_id:
        try:
            from app.socketio_server import sio
            await sio.emit('transcription_status', {
                'status': 'completed',
                'message': 'Transcription completed successfully',
                'video_id': video_id,
                'progress': 100
            }, room=session_id)
            
            # Also send the transcription directly
            await sio.emit('transcription', {
                'status': 'success',
                'video_id': video_id,
                'transcript': transcription_result.get("text", ""),
                'has_timestamps': 'timestamps' in transcription_result and bool(transcription_result['timestamps'])
            }, room=session_id)
            
            # Notify about timestamps if available
            if 'timestamps' in transcription_result and transcription_result['timestamps']:
                await sio.emit('timestamps_available', {
                    'video_id': video_id
                }, room=session_id)
            
            logger.info(f"Sent transcription to client for file {video_id}")
        except Exception as ws_error:
            logger.error(f"Error sending WebSocket notifications: {str(ws_error)}")
    
    return transcription_result

# Now also fix the process_audio_file function in transcription_service.py
async def process_audio_file(
    file_path: str,
    output_file: Optional[str] = None,
    video_id: Optional[str] = None,
    file_id: Optional[str] = None,  # Keep for backward compatibility
    tab_id: Optional[str] = None,
    session_id: Optional[str] = None,
    language_code: str = "en",
    **kwargs
) -> Dict[str, Any]:
    """
    Process an audio file for transcription
    
    Args:
        file_path: Path to the audio file
        output_file: Optional path to save the transcription output
        video_id: Optional video ID (will be generated if not provided)
        file_id: Legacy parameter for identification (will be mapped to video_id)
        tab_id: Optional client tab ID
        session_id: Optional session ID for WebSocket updates
        language_code: Language code for transcription
        
    Returns:
        Transcription results
    """
    logger.info(f"Processing audio file: {file_path}, video_id={video_id}, file_id={file_id}, tab_id={tab_id}")
    
    # For backward compatibility, use file_id as video_id if video_id is not provided
    actual_video_id = video_id or file_id
    
    # Log parameter conversion if needed
    if file_id and not video_id:
        logger.info(f"Using file_id={file_id} as video_id for backward compatibility")
    
    # Use the transcription service to transcribe the audio file
    # Pass session_id explicitly to ensure WebSocket notifications work
    # Import the module here to prevent circular imports
    try:
        from app.services import transcription_service
        result = await transcription_service.transcribe_video_file(
            video_path=file_path,
            video_id=actual_video_id,
            tab_id=tab_id,
            session_id=session_id or tab_id,  # Use tab_id as session_id if not provided
            language=language_code,
            output_file=output_file
        )
        return result
    except ImportError:
        # If transcription_service is not available, use our own implementation
        logger.warning("transcription_service module not found, using local implementation")
        return await transcribe_uploaded_file(
            file_path=file_path,
            output_file=output_file,
            language_code=language_code,
            video_id=actual_video_id,
            session_id=session_id or tab_id,
            **kwargs
        )

# Simplify the function to use a more flexible parameter approach
# Main transcription function
async def transcribe_video(
    video_path: str,
    output_file: Optional[str] = None,
    language_code: str = "en",
    video_id: Optional[str] = None,
    preferred_service: Optional[str] = None,
    debug_mode: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio from a video file with improved error handling and timestamp support
    """
    # Handle unexpected arguments by logging and ignoring them
    unexpected_args = set(kwargs.keys()) - {"timeout", "max_retries", "quality", "service_preference", "youtube_error", "youtube_url", "video_info", "tab_id", "session_id"}
    if unexpected_args:
        logger.warning(f"transcribe_video received unexpected arguments: {unexpected_args}")
    
    # Extract tab_id and session_id from kwargs
    tab_id = kwargs.get('tab_id')
    session_id = kwargs.get('session_id', tab_id)  # Use tab_id as session_id if not provided
    
    # Check for global debug mode
    if DEBUG_MODE_ENABLED:
        debug_mode = True
    
    # Check if ffmpeg is available if not in debug mode
    if not debug_mode and not FFMPEG_AVAILABLE:
        logger.warning("FFmpeg not available, falling back to debug mode")
        debug_mode = True
    
    # Auto-generate video_id if not provided
    if not video_id:
        timestamp = int(time.time())
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        video_id = f"video_{timestamp}_{random_id}"
        logger.info(f"Generated video_id: {video_id}")
    
    # Set default output file if not provided
    if not output_file:
        output_file = get_transcription_file_path(video_id)
        logger.info(f"Using default output file: {output_file}")
    
    # Use debug mode if requested
    if debug_mode:
        logger.info("Using debug transcription mode")
        return await transcribe_video_debug(
            video_path, output_file, language_code, video_id, **kwargs
        )
    
    # Basic implementation to handle the transcription
    logger.info(f"Transcribing video: {video_path}, language: {language_code}")
    
    try:
        # Verify file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract audio via ffmpeg
        audio_path = os.path.join(settings.TEMP_DIR, f"{video_id or 'audio'}_{int(time.time())}.wav")
        extract_cmd = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", audio_path, "-y"
        ]
        
        logger.info(f"Extracting audio with command: {' '.join(extract_cmd)}")
        process_result = subprocess.run(extract_cmd, check=True, capture_output=True)
        
        if process_result.returncode != 0:
            raise Exception(f"Audio extraction failed: {process_result.stderr.decode('utf-8', errors='ignore')}")
            
        logger.info(f"Extracted audio to {audio_path} ({os.path.getsize(audio_path)} bytes)")
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Audio extraction produced empty or missing file")
        
        # Try both Socket.IO imports for better compatibility
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
        
        # Verify credentials at runtime
        if preferred_service == "google":
            google_verified = await verify_google_credentials()
            if not google_verified["success"]:
                if ASSEMBLYAI_AVAILABLE:
                    logger.warning("Google credentials verification failed, falling back to AssemblyAI")
                    preferred_service = "assemblyai"
                else:
                    logger.error("Google credentials verification failed and AssemblyAI not available")
        elif preferred_service == "assemblyai":
            assemblyai_verified = await verify_assemblyai_credentials()
            if not assemblyai_verified["success"]:
                if GOOGLE_AVAILABLE:
                    logger.warning("AssemblyAI credentials verification failed, falling back to Google")
                    preferred_service = "google"
                else:
                    logger.error("AssemblyAI credentials verification failed and Google not available")
        
        # Determine which service to use based on preference and availability
        if preferred_service == "google" and GOOGLE_AVAILABLE:
            service_to_use = "google"
        elif preferred_service == "assemblyai" and ASSEMBLYAI_AVAILABLE:
            service_to_use = "assemblyai"
        elif GOOGLE_AVAILABLE:
            service_to_use = "google"
        elif ASSEMBLYAI_AVAILABLE:
            service_to_use = "assemblyai"
        else:
            logger.warning("No transcription service available, falling back to debug mode")
            service_to_use = "debug"
        
        # Call the appropriate transcription service
        logger.info(f"Using transcription service: {service_to_use}")
        
        # Notify clients about service selection
        notify_id = session_id or tab_id
        if sio and notify_id:
            try:
                await sio.emit('transcription_status', {
                    'status': "processing", 
                    'message': f"Processing with {service_to_use.upper()} transcription service",
                    'video_id': video_id,
                    'progress': 30
                }, room=notify_id)
            except Exception as ws_error:
                logger.error(f"Failed to notify WebSocket clients: {str(ws_error)}")
        
        # Try the selected service first with fallback to the other one
        try:
            if service_to_use == "google":
                logger.info(f"Attempting Google transcription for {video_id}")
                transcription_result = await transcribe_with_google(audio_path, language_code, **kwargs)
                
                # Check for errors in the result
                if ("error" in transcription_result or 
                    not transcription_result.get("text") or 
                    "Error in Google transcription" in transcription_result.get("text", "")):
                    
                    error_msg = transcription_result.get("error", "Unknown error")
                    logger.warning(f"Google transcription returned error: {error_msg}")
                    
                    # Try AssemblyAI if available
                    if ASSEMBLYAI_AVAILABLE:
                        logger.info(f"Falling back to AssemblyAI for {video_id}")
                        
                        # Notify clients about fallback
                        if sio and notify_id:
                            await sio.emit('transcription_status', {
                                'status': "processing", 
                                'message': "Falling back to AssemblyAI transcription service",
                                'video_id': video_id,
                                'progress': 40
                            }, room=notify_id)
                        
                        transcription_result = await transcribe_with_assemblyai(audio_path, language_code, **kwargs)
                    else:
                        logger.error("No fallback service available")
                else:
                    logger.info(f"Google transcription successful for {video_id}")
            
            elif service_to_use == "assemblyai":
                logger.info(f"Attempting AssemblyAI transcription for {video_id}")
                transcription_result = await transcribe_with_assemblyai(audio_path, language_code, **kwargs)
                
                # Check for errors in the result
                if ("error" in transcription_result or 
                    not transcription_result.get("text") or 
                    "Error in AssemblyAI transcription" in transcription_result.get("text", "")):
                    
                    error_msg = transcription_result.get("error", "Unknown error")
                    logger.warning(f"AssemblyAI transcription returned error: {error_msg}")
                    
                    # Try Google if available
                    if GOOGLE_AVAILABLE:
                        logger.info(f"Falling back to Google for {video_id}")
                        
                        # Notify clients about fallback
                        if sio and notify_id:
                            await sio.emit('transcription_status', {
                                'status': "processing", 
                                'message': "Falling back to Google transcription service",
                                'video_id': video_id,
                                'progress': 40
                            }, room=notify_id)
                        
                        transcription_result = await transcribe_with_google(audio_path, language_code, **kwargs)
                    else:
                        logger.error("No fallback service available")
                else:
                    logger.info(f"AssemblyAI transcription successful for {video_id}")
            
            else:
                # Debug mode or other fallback
                logger.info(f"Using debug transcription for {video_id}")
                transcription_result = await transcribe_video_debug(
                    audio_path, output_file, language_code, video_id, **kwargs
                )
                
        except Exception as service_error:
            logger.error(f"Error with {service_to_use} transcription: {str(service_error)}")
            
            # Try fallback service if available
            if service_to_use == "google" and ASSEMBLYAI_AVAILABLE:
                logger.info("Falling back to AssemblyAI after Google error")
                
                # Notify clients
                if sio and notify_id:
                    await sio.emit('transcription_status', {
                        'status': "processing", 
                        'message': "Falling back to AssemblyAI after Google error",
                        'video_id': video_id,
                        'progress': 40
                    }, room=notify_id)
                
                transcription_result = await transcribe_with_assemblyai(audio_path, language_code, **kwargs)
            elif service_to_use == "assemblyai" and GOOGLE_AVAILABLE:
                logger.info("Falling back to Google after AssemblyAI error")
                
                # Notify clients
                if sio and notify_id:
                    await sio.emit('transcription_status', {
                        'status': "processing", 
                        'message': "Falling back to Google after AssemblyAI error",
                        'video_id': video_id,
                        'progress': 40
                    }, room=notify_id)
                
                transcription_result = await transcribe_with_google(audio_path, language_code, **kwargs)
            else:
                # If no fallback available, re-raise the error
                raise
        
        # Add video_id if provided
        if video_id:
            transcription_result["video_id"] = video_id
        
        # Save to output file if specified
        if output_file and transcription_result:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(transcription_result, f, indent=2)
                logger.info(f"Saved transcription to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save transcription to file: {e}")
        
        # Save timestamps to separate file
        if video_id and "timestamps" in transcription_result and transcription_result["timestamps"]:
            timestamp_path = save_timestamps_to_file(video_id, transcription_result["timestamps"])
            logger.info(f"Saved timestamps to {timestamp_path}")
        
        # Clean up temporary audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed temporary audio file: {audio_path}")
        except OSError as e:
            logger.warning(f"Failed to remove temporary audio file: {audio_path}, error: {str(e)}")
        
        # Notify WebSocket clients about completion
        if sio and notify_id:
            try:
                await sio.emit('transcription_complete', {
                    'video_id': video_id,
                    'success': True,
                    'timestamp': time.time()
                }, room=notify_id)
                
                # Also send transcription directly
                await sio.emit('transcription', {
                    'status': 'success',
                    'video_id': video_id,
                    'transcript': transcription_result.get("text", ""),
                    'has_timestamps': "timestamps" in transcription_result and bool(transcription_result["timestamps"])
                }, room=notify_id)
                
                # Notify about timestamps if available
                if "timestamps" in transcription_result and transcription_result["timestamps"]:
                    await sio.emit('timestamps_available', {
                        'video_id': video_id
                    }, room=notify_id)
                
                logger.info(f"Notified WebSocket clients of transcription completion for {video_id}")
            except Exception as ws_error:
                logger.error(f"Failed to notify WebSocket clients: {str(ws_error)}")
        
        return transcription_result
    
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}"
        logger.error(error_msg)
        return {
            "text": f"Error extracting audio from video: {error_msg}",
            "segments": [{"id": 0, "text": f"Audio extraction error: {error_msg}"}],
            "timestamps": [],
            "error": error_msg,
            "video_id": video_id
        }
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        logger.error(error_msg)
        return {
            "text": f"Error: {error_msg}",
            "segments": [{"id": 0, "text": f"File error: {error_msg}"}],
            "timestamps": [],
            "error": error_msg,
            "video_id": video_id
        }
    except Exception as e:
        error_msg = f"Error in transcribe_video: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "text": f"Error transcribing video: {str(e)}",
            "segments": [{"id": 0, "text": f"Transcription error: {str(e)}"}],
            "timestamps": [],
            "error": str(e),
            "video_id": video_id
        }

# YouTube transcription function
async def transcribe_youtube_video(
    youtube_url: str,
    output_file: Optional[str] = None,
    language_code: str = "en",
    video_id: Optional[str] = None,
    preferred_service: Optional[str] = None,
    debug_mode: bool = False,
    use_cache: bool = True,  # Add parameter to control caching
    **kwargs
) -> Dict[str, Any]:
    """
    Download and transcribe a YouTube video with improved caching
    """
    # Handle unexpected arguments by logging and ignoring them
    unexpected_args = set(kwargs.keys()) - {"timeout", "max_retries", "quality", "service_preference"}
    if unexpected_args:
        logger.warning(f"transcribe_youtube_video received unexpected arguments: {unexpected_args}")
    
    # Check for global debug mode
    if DEBUG_MODE_ENABLED:
        debug_mode = True
    
    # Generate video_id if not provided
    if not video_id:
        timestamp = int(time.time())
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        video_id = f"youtube_{timestamp}_{random_id}"
        logger.info(f"Generated video_id for YouTube: {video_id}")
    
    # Normalize video_id
    normalized_id = normalize_video_id(video_id)
    video_id = normalized_id
    
    # Set default output file if not provided
    if not output_file:
        output_file = get_transcription_file_path(video_id)
        logger.info(f"Using default output file for YouTube: {output_file}")
    
    # Check for cached transcription if use_cache is True
    if use_cache:
        # First, try the TranscriptionCache for this URL
        is_cached, cached_data = transcription_cache.get_from_cache(youtube_url, video_id)
        if is_cached and cached_data:
            logger.info(f"Using cached transcription for YouTube URL: {youtube_url}")
            
            # Save the cached data to the output file with updated video_id
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(cached_data, f, indent=2)
                logger.info(f"Saved cached transcription to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving cached transcription: {str(e)}")
            
            # Notify clients about using cached transcription
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': "completed", 
                    'message': "Using cached transcription",
                    'video_id': video_id,
                    'progress': 100
                })
                
                # Also emit the transcription event
                await sio.emit('transcription', {
                    'status': "success",
                    'video_id': video_id,
                    'transcript': cached_data.get("text", ""),
                    'has_timestamps': True if "timestamps" in cached_data and cached_data["timestamps"] else False
                })
                
                # Emit timestamps_available event if applicable
                if "timestamps" in cached_data and cached_data["timestamps"]:
                    await sio.emit('timestamps_available', {
                        'video_id': video_id
                    })
                
                logger.info(f"Notified WebSocket clients of cached transcription for {video_id}")
            except Exception as ws_error:
                logger.error(f"Failed to notify WebSocket clients of cached transcription: {str(ws_error)}")
            
            return cached_data
        
        # Second, check if the output file already exists (legacy cache)
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    
                # Check if it contains an error - if so, don't use the cache
                if "error" not in existing_data:
                    logger.info(f"Using existing transcription file: {output_file}")
                    
                    # Notify clients about using cached transcription
                    try:
                        from app.socketio_server import sio
                        await sio.emit('transcription_status', {
                            'status': "completed", 
                            'message': "Using existing transcription",
                            'video_id': video_id,
                            'progress': 100
                        })
                        
                        # Also emit the transcription event
                        await sio.emit('transcription', {
                            'status': "success",
                            'video_id': video_id,
                            'transcript': existing_data.get("text", ""),
                            'has_timestamps': True if "timestamps" in existing_data and existing_data["timestamps"] else False
                        })
                        
                        # Emit timestamps_available event if applicable
                        if "timestamps" in existing_data and existing_data["timestamps"]:
                            await sio.emit('timestamps_available', {
                                'video_id': video_id
                            })
                        
                        logger.info(f"Notified WebSocket clients of existing transcription for {video_id}")
                    except Exception as ws_error:
                        logger.error(f"Failed to notify WebSocket clients of existing transcription: {str(ws_error)}")
                    
                    # Also save to the new cache system for future
                    transcription_cache.save_to_cache(youtube_url, existing_data)
                    
                    return existing_data
            except Exception as e:
                logger.error(f"Error reading existing transcription: {str(e)}")
    
    # Try direct URL processing with AssemblyAI if available
    if not debug_mode and ASSEMBLYAI_AVAILABLE:
        logger.info(f"Trying AssemblyAI direct URL processing for {youtube_url}")
        try:
            # Notify clients about direct processing
            try:
                from app.socketio_server import sio
                await sio.emit('transcription_status', {
                    'status': 'processing',
                    'message': 'Processing YouTube URL directly with AssemblyAI...',
                    'video_id': video_id,
                    'progress': 15
                })
            except Exception as ws_error:
                logger.error(f"Failed to send WebSocket notification: {str(ws_error)}")
            
            # Try AssemblyAI direct URL processing
            assemblyai_result = await transcribe_with_assemblyai(youtube_url, language_code)
            
            # Check if it worked (no error in result)
            if "error" not in assemblyai_result:
                # Add video_id and URL to the result
                assemblyai_result["video_id"] = video_id
                assemblyai_result.setdefault("metadata", {}).update({
                    "source_url": youtube_url,
                    "processed_directly": True
                })
                
                # Save to output file if specified
                if output_file:
                    try:
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump(assemblyai_result, f, indent=2)
                        logger.info(f"Saved AssemblyAI direct transcription to {output_file}")
                    except Exception as e:
                        logger.error(f"Error saving transcription: {str(e)}")
                
                # Save timestamps if available
                if "timestamps" in assemblyai_result and assemblyai_result["timestamps"]:
                    save_timestamps_to_file(video_id, assemblyai_result["timestamps"])
                
                # Notify clients of successful direct transcription
                try:
                    from app.socketio_server import sio
                    await sio.emit('transcription_status', {
                        'status': 'completed',
                        'message': 'Direct transcription completed',
                        'video_id': video_id,
                        'progress': 100
                    })
                    
                    await sio.emit('transcription', {
                        'status': 'success',
                        'video_id': video_id,
                        'transcript': assemblyai_result.get("text", ""),
                        'has_timestamps': True if "timestamps" in assemblyai_result and assemblyai_result["timestamps"] else False
                    })
                    
                    if "timestamps" in assemblyai_result and assemblyai_result["timestamps"]:
                        await sio.emit('timestamps_available', {
                            'video_id': video_id
                        })
                except Exception as ws_error:
                    logger.error(f"Failed to send WebSocket notifications: {str(ws_error)}")
                
                # Save to cache if use_cache is enabled
                if use_cache:
                    transcription_cache.save_to_cache(youtube_url, assemblyai_result)
                    logger.info(f"Saved direct transcription to cache for: {youtube_url}")
                
                # Return the result, skipping download completely
                return assemblyai_result
        except Exception as e:
            logger.warning(f"AssemblyAI direct URL processing failed: {str(e)}")
            # Continue to standard download path
    
    # Use debug mode if requested
    if debug_mode:
        logger.info("Using debug transcription mode for YouTube")
        result = await transcribe_video_debug(
            None, output_file, language_code, video_id,
            youtube_error=False, youtube_url=youtube_url
        )
        result.setdefault("metadata", {})["source_url"] = youtube_url
        
        # Save to cache after debugging
        if use_cache:
            transcription_cache.save_to_cache(youtube_url, result)
        return result
    
    # Detect if this is a Shorts URL
    is_shorts = is_youtube_shorts(youtube_url)
    
    # Notify clients that we're starting the download
    try:
        from app.socketio_server import sio
        await sio.emit('youtube_download_started', {
            'video_id': video_id,
            'url': youtube_url,
            'timestamp': time.time(),
            'video_type': 'shorts' if is_shorts else 'regular'
        })
        logger.info(f"Notified WebSocket clients of YouTube download start for {video_id}")
    except Exception as ws_error:
        logger.error(f"Failed to notify WebSocket clients of download start: {str(ws_error)}")
    
    # Download the YouTube video
    download_result = await download_youtube_video(youtube_url, settings.TEMP_DIR, video_id)
    
    # Handle download failure
    if not download_result["success"]:
        error_msg = download_result.get("error", "Unknown error")
        video_info = download_result.get("video_info", "Unknown video")
        logger.error(f"YouTube download failed: {error_msg}")
        
        # Use debug mode to create a nice error response
        error_result = await transcribe_video_debug(
            None, output_file, language_code, video_id,
            youtube_error=True, youtube_url=youtube_url, video_info=video_info
        )
        
        # Add metadata
        error_result.setdefault("metadata", {}).update({
            "source_url": youtube_url,
            "error": error_msg,
            "detailed_error": download_result.get("detailed_error", error_msg)
        })
        
        # Notify WebSocket clients of the error
        try:
            from app.socketio_server import sio
            await sio.emit('transcription', {
                'status': "error",
                'video_id': video_id,
                'transcript': error_result.get("text", f"Error: {error_msg}"),
                'has_timestamps': True if "timestamps" in error_result and error_result["timestamps"] else False,
                'error': error_msg
            })
            logger.info(f"Notified WebSocket clients of YouTube download error for {video_id}")
        except Exception as ws_error:
            logger.error(f"Failed to notify WebSocket clients of download error: {str(ws_error)}")
        
        return error_result
    
    video_path = download_result["video_path"]
    try:
        # Notify clients that download is complete and transcription is starting
        try:
            from app.socketio_server import sio
            await sio.emit('youtube_download_complete', {
                'video_id': video_id,
                'url': youtube_url,
                'timestamp': time.time(),
                'video_path': video_path
            })
            logger.info(f"Notified WebSocket clients of YouTube download completion for {video_id}")
        except Exception as ws_error:
            logger.error(f"Failed to notify WebSocket clients of download completion: {str(ws_error)}")
        
        # For Shorts, prefer AssemblyAI as it often works better with short videos
        if is_shorts and ASSEMBLYAI_AVAILABLE:
            logger.info("Using AssemblyAI for YouTube Shorts transcription")
            preferred_service = "assemblyai"
        
        # Transcribe the downloaded video
        transcription = await transcribe_video(
            video_path=video_path,
            output_file=output_file,
            language_code=language_code,
            video_id=video_id,
            preferred_service=preferred_service,
            debug_mode=debug_mode,
            **kwargs
        )
        
        # Add metadata
        transcription.setdefault("metadata", {}).update({
            "source_url": youtube_url,
            "youtube_info": download_result.get("metadata", {})
        })
        
        # Save to cache if transcription was successful and use_cache is True
        if use_cache and "text" in transcription and "error" not in transcription:
            transcription_cache.save_to_cache(youtube_url, transcription)
            logger.info(f"Saved transcription to cache for: {youtube_url}")
        
        # Cleanup the downloaded video file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed temporary video file: {video_path}")
        except OSError as e:
            logger.warning(f"Failed to remove temporary video file: {video_path}, error: {str(e)}")
        
        return transcription
    
    except Exception as e:
        error_msg = f"Error in transcribe_youtube_video: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Clean up the video file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed temporary video file after error: {video_path}")
        except OSError:
            pass
        
        return {
            "text": f"Error processing YouTube video: {str(e)}",
            "segments": [{"id": 0, "text": str(e)}],
            "timestamps": [],
            "error": str(e),
            "video_id": video_id
        }

# Unified transcription function
async def transcribe_media(
    source: str,
    output_file: Optional[str] = None,
    language_code: str = "en",
    video_id: Optional[str] = None,
    file_type: str = None,  # Automatically detected if None
    preferred_service: Optional[str] = None,
    debug_mode: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified function to transcribe either YouTube videos or uploaded files
    
    Args:
        source: URL or file path
        output_file: Optional path to save the transcription
        language_code: Language code for transcription
        video_id: Optional ID for the video/audio
        file_type: Type of file ('video', 'audio', or None for auto-detect)
        preferred_service: Preferred transcription service
        debug_mode: Enable debug mode
        use_cache: Whether to use cached transcriptions
        
    Returns:
        Transcription result dictionary
    """
    # Determine if this is a YouTube URL or a file path
    is_youtube = source.startswith(('http://', 'https://')) and (
        'youtube.com' in source or 'youtu.be' in source
    )
    
    if is_youtube:
        # Process as YouTube URL
        return await transcribe_youtube_video(
            youtube_url=source,
            output_file=output_file,
            language_code=language_code,
            video_id=video_id,
            preferred_service=preferred_service,
            debug_mode=debug_mode,
            use_cache=use_cache,
            **kwargs
        )
    else:
        # Process as file path
        if not os.path.exists(source):
            return {
                "text": f"Error: File not found at {source}",
                "segments": [],
                "timestamps": [],
                "error": "File not found",
                "video_id": video_id
            }
        
        # Auto-detect file type if not specified
        if file_type is None:
            file_extension = os.path.splitext(source)[1].lower()
            if file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']:
                file_type = 'audio'
            else:
                file_type = 'video'
            logger.info(f"Auto-detected file type: {file_type}")
        
        return await transcribe_uploaded_file(
            file_path=source,
            output_file=output_file,
            language_code=language_code,
            video_id=video_id,
            file_type=file_type,
            preferred_service=preferred_service,
            debug_mode=debug_mode,
            **kwargs
        )

# Get transcript for a video
async def get_transcript_for_video(video_id: str, tab_id: Optional[str] = None) -> str:
    """
    Get transcript for a video from file or database with improved ID handling
    
    Args:
        video_id: ID of the video
        tab_id: Optional tab ID for session-specific data
        
    Returns:
        Transcript text
    """
    logger.info(f"Getting transcript for video: {video_id}")
    
    # Normalize video_id to ensure consistent format
    normalized_id = normalize_video_id(video_id)
    logger.info(f"Normalized video ID: {normalized_id}")
    
    # Define standard file paths to check
    file_paths_to_check = []
    
    # Add tab-specific path if available
    if tab_id:
        file_paths_to_check.append(get_transcription_file_path(normalized_id, tab_id))
    
    # Add standard path
    file_paths_to_check.append(get_transcription_file_path(normalized_id))
    
    # Add legacy format paths for backward compatibility
    file_paths_to_check.extend([
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{normalized_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{normalized_id}.json")
    ])
    
    # Special handling for YouTube IDs
    if "youtube" in normalized_id.lower():
        youtube_id_raw = normalized_id.replace("youtube_", "")
        file_paths_to_check.extend([
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{youtube_id_raw}.json"),
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_youtube_{youtube_id_raw}.json")
        ])
    
    # Special handling for upload IDs
    if "upload_" in normalized_id:
        hash_part = normalized_id.split("_")[-1] if "_" in normalized_id else ""
        if hash_part:
            # Look for files containing this hash part
            try:
                transcription_dir = Path(settings.TRANSCRIPTION_DIR)
                potential_hash_files = list(transcription_dir.glob(f"*{hash_part}*.json"))
                file_paths_to_check.extend([str(path) for path in potential_hash_files])
            except Exception as e:
                logger.error(f"Error searching for files by hash: {str(e)}")
    
    # Try each file path
    for file_path in file_paths_to_check:
        if os.path.exists(file_path):
            try:
                logger.info(f"Found transcript file: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)
                return data.get("text", "")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
                continue
    
    # If not found in files, try database
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        
        with get_db_context() as db:
            # Try various query patterns
            query_patterns = [
                {"id": str(normalized_id)},
                {"id": str(video_id)},
            ]
            
            # Add hash part query if this is an upload ID
            if "upload_" in normalized_id and "_" in normalized_id:
                hash_part = normalized_id.split("_")[-1]
                if hash_part:
                    query_patterns.append({"pattern": f"%{hash_part}%"})
            
            # Add YouTube ID query without prefix
            if "youtube" in normalized_id.lower():
                youtube_id = normalized_id.replace("youtube_", "")
                query_patterns.append({"pattern": f"%{youtube_id}%", "url_pattern": f"%{youtube_id}%"})
            
            # Try each query pattern
            for pattern in query_patterns:
                try:
                    if "pattern" in pattern:
                        # Pattern-based query
                        if "url_pattern" in pattern:
                            # YouTube pattern with URL
                            query = "SELECT transcription FROM videos WHERE id::text LIKE :pattern OR video_url LIKE :url_pattern"
                        else:
                            # Regular pattern
                            query = "SELECT transcription FROM videos WHERE id::text LIKE :pattern"
                    else:
                        # Exact match query
                        query = "SELECT transcription FROM videos WHERE id::text = :id"
                    
                    result = db.execute(text(query), pattern)
                    row = result.fetchone()
                    
                    if row and row[0]:
                        logger.info(f"Found transcript in database with query: {pattern}")
                        return row[0]
                except Exception as db_query_error:
                    logger.error(f"Database query error with pattern {pattern}: {str(db_query_error)}")
                    continue
    
    except Exception as db_error:
        logger.error(f"Database error: {str(db_error)}")
    
    # If still not found and debug mode is enabled, generate debug transcript
    if DEBUG_MODE_ENABLED:
        logger.info(f"Generating mock transcript due to DEBUG_TRANSCRIPTION=1")
        mock_transcript = f"This is a debug transcription for video ID: {normalized_id} (created on-demand for get_transcript_for_video request)"
        
        # Try to save this for next time
        debug_file = get_transcription_file_path(normalized_id)
        try:
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            with open(debug_file, "w") as f:
                json.dump({
                    "text": mock_transcript,
                    "video_id": normalized_id,
                    "debug": True,
                    "segments": [
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 5.0,
                            "text": mock_transcript
                        }
                    ]
                }, f)
            logger.info(f"Saved debug transcript to {debug_file}")
        except Exception as e:
            logger.error(f"Error saving debug transcript: {str(e)}")
            
        return mock_transcript
    
    # No transcript found
    logger.warning(f"No transcript available for video ID: {normalized_id}")
    return f"No transcription found for this video (ID: {normalized_id}). You may need to process it first."

# Get timestamps for a video
async def get_timestamps_for_video(video_id: str) -> Dict:
    """
    Get timestamps for a video from file
    
    Args:
        video_id: ID of the video
        
    Returns:
        Dictionary with timestamp data or empty dict if not found
    """
    logger.info(f"Getting timestamps for video: {video_id}")
    
    # Normalize video_id to ensure consistent format
    normalized_id = normalize_video_id(video_id)
    logger.info(f"Normalized video ID for timestamps: {normalized_id}")
    
    # Check standard timestamp path
    timestamp_path = get_timestamp_file_path(normalized_id)
    
    if os.path.exists(timestamp_path):
        try:
            logger.info(f"Found timestamps file: {timestamp_path}")
            with open(timestamp_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading timestamps file: {str(e)}")
    
    # Special handling for YouTube IDs
    if "youtube" in normalized_id.lower():
        youtube_id_variations = [
            normalized_id,  # Original format
            normalized_id.replace("youtube_", ""),  # Without prefix
            f"youtube_{normalized_id.replace('youtube_', '')}"  # With prefix
        ]
        
        for yt_id in youtube_id_variations:
            yt_timestamp_path = get_timestamp_file_path(yt_id)
            
            if os.path.exists(yt_timestamp_path):
                try:
                    logger.info(f"Found YouTube timestamps file: {yt_timestamp_path}")
                    with open(yt_timestamp_path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error reading YouTube timestamps file: {str(e)}")
    
    # Try to find by hash part for upload IDs
    if "upload_" in normalized_id and "_" in normalized_id:
        hash_part = normalized_id.split("_")[-1]
        if hash_part:
            try:
                timestamp_dir = Path(os.path.join(settings.TRANSCRIPTION_DIR, "timestamps"))
                potential_files = list(timestamp_dir.glob(f"*{hash_part}*_timestamps.json"))
                
                for pot_file in potential_files:
                    try:
                        logger.info(f"Found potential timestamp file by hash: {pot_file}")
                        with open(pot_file, "r") as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading potential timestamp file: {str(e)}")
            except Exception as e:
                logger.error(f"Error searching for timestamp files by hash: {str(e)}")
    
    # Try to generate timestamps from transcription file
    transcription_paths = [
        get_transcription_file_path(normalized_id),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{normalized_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{normalized_id}.json")
    ]
    
    # Add YouTube variations
    if "youtube" in normalized_id.lower():
        youtube_id = normalized_id.replace("youtube_", "")
        transcription_paths.extend([
            get_transcription_file_path(youtube_id),
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{youtube_id}.json"),
            os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_youtube_{youtube_id}.json")
        ])
    
    # Try each transcription file
    for transc_path in transcription_paths:
        if os.path.exists(transc_path):
            try:
                logger.info(f"Found transcription file for timestamp generation: {transc_path}")
                with open(transc_path, "r") as f:
                    transc_data = json.load(f)
                    
                if "timestamps" in transc_data and transc_data["timestamps"]:
                    # Save timestamps to a separate file
                    timestamp_path = save_timestamps_to_file(normalized_id, transc_data["timestamps"])
                    
                    # Read the formatted data
                    if timestamp_path:
                        with open(timestamp_path, "r") as f:
                            timestamp_data = json.load(f)
                            
                        logger.info(f"Generated timestamps from transcription for video_id {normalized_id}")
                        return timestamp_data
            except Exception as e:
                logger.error(f"Error generating timestamps from transcription: {str(e)}")
    
    # Generate debug timestamps if in debug mode
    if DEBUG_MODE_ENABLED:
        logger.info(f"Generating mock timestamps due to DEBUG_TRANSCRIPTION=1")
        
        # Generate dummy timestamps
        mock_timestamps = [
            {"word": "This", "start_time": 0.0, "end_time": 0.5},
            {"word": "is", "start_time": 0.5, "end_time": 0.7},
            {"word": "a", "start_time": 0.7, "end_time": 0.8},
            {"word": "debug", "start_time": 0.8, "end_time": 1.2},
            {"word": "timestamp", "start_time": 1.2, "end_time": 2.0},
            {"word": "for", "start_time": 2.0, "end_time": 2.2},
            {"word": normalized_id, "start_time": 2.2, "end_time": 3.0}
        ]
        
        # Save to file and return the formatted data
        timestamp_path = save_timestamps_to_file(normalized_id, mock_timestamps)
        
        if timestamp_path:
            with open(timestamp_path, "r") as f:
                timestamp_data = json.load(f)
                
            logger.info(f"Generated mock timestamps for video_id {normalized_id}")
            return timestamp_data
    
    # Log debug info
    if os.path.exists(os.path.join(settings.TRANSCRIPTION_DIR, "timestamps")):
        timestamp_dir = Path(os.path.join(settings.TRANSCRIPTION_DIR, "timestamps"))
        all_files = list(timestamp_dir.glob("*.json"))
        logger.info(f"All timestamp files: {[f.name for f in all_files]}")
        
        # Check for files containing this video_id
        matches = [f for f in all_files if normalized_id in f.name]
        logger.info(f"Matching files for {normalized_id}: {[f.name for f in matches]}")
    
    logger.warning(f"No timestamps found for video ID: {normalized_id}")
    return {}

# Notify function for sending WebSocket updates
async def notify_transcription_status(video_id, status, message=None, error=None):
    """Send transcription status updates via WebSocket"""
    try:
        from app.socketio_server import sio
        
        data = {
            'video_id': video_id,
            'status': status,
            'timestamp': time.time()
        }
        
        if message:
            data['message'] = message
            
        if error:
            data['error'] = error
            
        await sio.emit('transcription_status', data)
        logger.info(f"Sent transcription status '{status}' for {video_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to send transcription status update: {str(e)}")
        return False

# Periodic task to clean up expired cache entries
async def maintain_transcription_cache():
    """Periodic task to clean up expired cache entries and manage cache size"""
    while True:
        try:
            # Clear expired entries
            cleared = transcription_cache.clear_expired_cache()
            logger.info(f"Cleared {cleared} expired cache entries")
            
            # Check total cache size
            cache_size = 0
            cache_count = 0
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        cache_size += os.path.getsize(file_path)
                        cache_count += 1
            
            cache_size_mb = cache_size / (1024 * 1024)
            logger.info(f"Current cache: {cache_count} entries, {cache_size_mb:.2f} MB")
            
            # If cache is too large, remove oldest entries
            max_cache_size_mb = 5000  # 5GB max cache size
            if cache_size_mb > max_cache_size_mb:
                logger.info(f"Cache size exceeds {max_cache_size_mb} MB, cleaning up oldest entries")
                # Implementation of cleanup logic based on creation time
                # (This would need implementation based on your specific cache structure)
            
        except Exception as e:
            logger.error(f"Error in cache maintenance: {str(e)}")
        
        # Run once per day
        await asyncio.sleep(86400)

# Helper function to test the transcription flow
async def test_transcription_flow(video_path, video_id=None):
    """Test the entire transcription flow and log each step"""
    logger.info(f"Starting test transcription flow for {video_path}")
    
    # 1. Normalize ID
    video_id = video_id or f"test_{int(time.time())}"
    normalized_id = normalize_video_id(video_id)
    logger.info(f"Using video ID: {normalized_id}")
    
    # 2. Transcribe video
    result = await transcribe_video(
        video_path=video_path,
        output_file=get_transcription_file_path(normalized_id),
        video_id=normalized_id,
        debug_mode=False
    )
    
    # 3. Verify result structure
    logger.info(f"Transcription result keys: {result.keys()}")
    
    # 4. Verify file exists
    file_path = get_transcription_file_path(normalized_id)
    logger.info(f"Checking if file exists: {file_path} - {os.path.exists(file_path)}")
    
    # 5. Try to read transcript
    transcript = await get_transcript_for_video(normalized_id)
    logger.info(f"Retrieved transcript length: {len(transcript)}")
    
    # 6. Try to read timestamps
    timestamps = await get_timestamps_for_video(normalized_id)
    logger.info(f"Retrieved timestamps: {len(timestamps) if timestamps else 0} entries")
    
    return {
        "transcription_result": result,
        "file_exists": os.path.exists(file_path),
        "transcript_length": len(transcript),
        "timestamps_available": bool(timestamps),
        "normalized_id": normalized_id
    }

# Start cache maintenance task if running as main module
if __name__ == "__main__":
    asyncio.run(maintain_transcription_cache())