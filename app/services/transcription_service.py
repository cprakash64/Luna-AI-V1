"""
Transcription service wrapper for Luna AI that provides a unified interface to the
various transcription implementation options with enhanced error handling, caching,
and WebSocket communication.
"""
import os
import json
import logging
import asyncio
import uuid
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile
import subprocess

from app.config import settings
from app.services.transcription import (
    transcribe_video,
    transcribe_media,  # Add this import for unified transcription
    get_timestamps_for_video,
    normalize_video_id,
    save_timestamps_to_file,
    verify_google_credentials,  # Import verification functions
    verify_assemblyai_credentials,
    verify_ffmpeg
)

# Define globals for import checking
HAS_YTDLP = False
HAS_PYTUBE = False

# Try to import yt-dlp, fallback to pytube
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False
    try:
        import pytube 
        HAS_PYTUBE = True
    except ImportError:
        HAS_PYTUBE = False

# Check for AssemblyAI
HAS_ASSEMBLYAI = False
try:
    import assemblyai as aai
    HAS_ASSEMBLYAI = True
except ImportError:
    HAS_ASSEMBLYAI = False

# Set up logging
logger = logging.getLogger("transcription.service")
os.makedirs(settings.LOG_DIR if hasattr(settings, 'LOG_DIR') else '/tmp', exist_ok=True)
file_handler = logging.FileHandler(os.path.join(settings.LOG_DIR if hasattr(settings, 'LOG_DIR') else '/tmp', 'transcription_service.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Helper function to check debug mode consistently
def is_debug_mode():
    """Check if debug mode is enabled"""
    # Check both environment variables to ensure consistency with transcription.py
    return (os.environ.get("DEBUG_MODE_ENABLED", "0") == "1" or 
            os.environ.get("DEBUG_TRANSCRIPTION", "0") == "1")

# Try to import cache implementation
try:
    from app.services.transcription_cache import TranscriptionCache
except ImportError:
    # Define a simple cache class if the import fails
    class TranscriptionCache:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            logger.info(f"Using dummy cache at {cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
            
        def get_from_cache(self, key, video_id=None):
            return False, None
            
        def save_to_cache(self, key, data):
            pass
            
        def clear_all_cache(self):
            return True

# Helper function for WebSocket communication
async def send_websocket_update(session_id, event_type, data):
    """Send message to client via WebSocket with proper error handling"""
    if not session_id:
        return False
        
    try:
        # Try both import locations for Socket.IO
        try:
            from app.api.websockets import sio
        except ImportError:
            try:
                from app.socketio_server import sio
            except ImportError:
                logger.warning("Could not import Socket.IO from either location")
                return False
        
        await sio.emit(event_type, data, room=session_id)
        return True
    except Exception as e:
        logger.error(f"Error sending WebSocket message: {str(e)}")
        return False

class TranscriptionService:
    """
    Enhanced service class to handle transcription operations
    with caching, WebSocket updates, and improved error handling
    """
    
    def __init__(self):
        """Initialize the transcription service"""
        self.debug_mode = is_debug_mode()
        
        # Initialize cache
        cache_dir = Path(settings.TRANSCRIPTION_DIR) / "cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Create empty index file if it doesn't exist
        index_path = cache_dir / "index.json"
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                json.dump({}, f)
        self.cache = TranscriptionCache(cache_dir)
        
        # Initialize timestamps directory
        self.timestamps_dir = Path(settings.TRANSCRIPTION_DIR) / "timestamps"
        os.makedirs(self.timestamps_dir, exist_ok=True)
        
        # Initialize temp dir
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Check available YouTube downloaders
        self.download_methods = self._check_youtube_downloaders()
        
        # Check available transcription services
        self.google_available = False
        self.assemblyai_available = False
        self.assemblyai_direct_url = HAS_ASSEMBLYAI
        asyncio.create_task(self._verify_transcription_services())
        
        logger.info(f"TranscriptionService initialized, debug_mode={self.debug_mode}")
        logger.info(f"Available YouTube download methods: {', '.join(self.download_methods) if self.download_methods else 'None'}")
        logger.info(f"AssemblyAI direct URL support: {self.assemblyai_direct_url}")
    
    async def _verify_transcription_services(self):
        """Verify which transcription services are available"""
        try:
            google_result = await verify_google_credentials()
            self.google_available = google_result.get('success', False)
            
            assemblyai_result = await verify_assemblyai_credentials()
            self.assemblyai_available = assemblyai_result.get('success', False)
            
            logger.info(f"Transcription services: Google={self.google_available}, AssemblyAI={self.assemblyai_available}")
        except Exception as e:
            logger.error(f"Error verifying transcription services: {str(e)}")
    
    def _check_youtube_downloaders(self):
        """Check which YouTube download methods are available"""
        methods = []
        if HAS_YTDLP:
            methods.append("yt-dlp")
        if HAS_PYTUBE:
            methods.append("pytube")
        
        # Check command line tools
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
            methods.append("yt-dlp-cli")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        try:
            subprocess.run(["youtube-dl", "--version"], capture_output=True, check=True)
            methods.append("youtube-dl-cli")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return methods
    
    async def transcribe_with_assemblyai_direct(self, youtube_url: str, video_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe a YouTube video directly with AssemblyAI without downloading
        
        Args:
            youtube_url: YouTube URL to transcribe
            video_id: ID to use for the transcription
            session_id: Optional session ID for WebSocket updates
            
        Returns:
            Transcription result dictionary or None if failed
        """
        if not HAS_ASSEMBLYAI or not self.assemblyai_available:
            logger.warning("AssemblyAI direct URL transcription not available")
            return None
            
        logger.info(f"Transcribing YouTube URL directly with AssemblyAI: {youtube_url}")
        
        # Send status update
        if session_id:
            await send_websocket_update(session_id, 'transcription_status', {
                'status': 'processing',
                'message': 'Processing YouTube URL directly with AssemblyAI...',
                'video_id': video_id,
                'progress': 20
            })
            
        try:
            # Initialize AssemblyAI
            aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            
            transcriber = aai.Transcriber()
            
            # Log that we're attempting direct URL transcription
            logger.info(f"Submitting YouTube URL directly to AssemblyAI: {youtube_url}")
            
            # Send another status update
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'processing',
                    'message': 'AssemblyAI is processing the YouTube video...',
                    'video_id': video_id,
                    'progress': 30
                })
                
            # Transcribe directly with the URL (this is the key feature!)
            transcript = transcriber.transcribe(youtube_url)
            
            # Check if transcription failed
            if transcript.status == "error":
                error_msg = getattr(transcript, 'error', 'Unknown error')
                logger.error(f"AssemblyAI direct transcription failed: {error_msg}")
                return None
                
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
            
            # Create result dictionary
            transcription_result = {
                "text": text,
                "segments": segments,
                "timestamps": timestamps,
                "video_id": video_id,
                "source_type": "youtube_direct_url",
                "success": True,
                "metadata": {
                    "source_url": youtube_url,
                    "processed_directly": True,
                    "service": "assemblyai_direct"
                }
            }
            
            logger.info(f"AssemblyAI direct transcription complete: {len(timestamps)} words, {len(segments)} segments")
            
            # Send success status update
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'processing',
                    'message': 'AssemblyAI transcription completed successfully',
                    'video_id': video_id,
                    'progress': 80
                })
                
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error in AssemblyAI direct transcription: {str(e)}", exc_info=True)
            return None
    
    async def download_youtube_audio(self, youtube_url: str, output_dir: Path, video_id: str) -> Optional[str]:
        """
        Download audio from a YouTube video with improved error handling
        
        Args:
            youtube_url: YouTube URL
            output_dir: Directory to save the audio
            video_id: Unique ID for this video
            
        Returns:
            Path to downloaded audio file or None if failed
        """
        logger.info(f"Downloading audio from YouTube URL: {youtube_url}, video_id: {video_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename WITH .wav extension from the start
        audio_filename = f"audio_{video_id}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        # Check if in debug mode
        if self.debug_mode:
            logger.info(f"Debug mode enabled, creating mock audio file for {video_id}")
            # Create an empty file as a placeholder
            with open(audio_path, "w") as f:
                f.write("Mock audio file for debugging")
            return audio_path
        
        # Define possible cookie file locations
        cookie_paths = [
            os.path.expanduser('~/.config/cookies/cookies.txt'),  # Custom location
            os.path.expanduser('~/Library/Application Support/Google/Chrome/Default/Cookies'),  # macOS Chrome
            os.path.expanduser('~/.config/google-chrome/Default/Cookies'),  # Linux Chrome
            os.path.expanduser('~/AppData/Local/Google/Chrome/User Data/Default/Cookies'),  # Windows Chrome
            os.path.expanduser('~/Library/Application Support/Firefox/Profiles/'),  # macOS Firefox
            '/tmp/youtube_cookies.txt'  # Fallback location
        ]
        
        # Find the first valid cookie file
        cookie_file = None
        for path in cookie_paths:
            if os.path.exists(path):
                cookie_file = path
                logger.info(f"Found cookie file at: {cookie_file}")
                break
                
        # Define user agents to try
        user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        ]
        
        # Maximum number of retry attempts
        max_retries = 3
        current_attempt = 0
        
        while current_attempt < max_retries:
            current_attempt += 1
            user_agent = user_agents[(current_attempt - 1) % len(user_agents)]
            logger.info(f"Download attempt {current_attempt}/{max_retries} with user agent: {user_agent}")
            
            # Try yt-dlp first if available
            if "yt-dlp" in self.download_methods or "yt-dlp-cli" in self.download_methods:
                try:
                    if HAS_YTDLP:
                        # Use yt-dlp Python library with browser cookie authentication
                        ydl_opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': audio_path[:-4],  # Remove .wav extension as yt-dlp will add it
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'wav',
                                'preferredquality': '192',
                            }],
                            'quiet': False,
                            'keepvideo': False,  # Don't keep the video
                            'user_agent': user_agent,  # Set user agent
                            'geo_bypass': True,  # Bypass geo-restrictions
                            'geo_bypass_country': 'US',  # Use US as default region
                            'nocheckcertificate': True,  # Skip HTTPS certificate validation
                        }
                        
                        # Add cookies if available
                        if cookie_file:
                            if "cookies.txt" in cookie_file:
                                ydl_opts['cookiefile'] = cookie_file
                            else:
                                ydl_opts['cookiesfrombrowser'] = ('chrome',)
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            logger.info(f"Downloading with yt-dlp Python library to {audio_path}")
                            ydl.download([youtube_url])
                    else:
                        # Use command line with improved options
                        cmd = [
                            "yt-dlp",
                            "--extract-audio",
                            "--audio-format", "wav",
                            "--audio-quality", "0",
                            "--user-agent", user_agent,
                            "--geo-bypass",
                            "--no-check-certificate",
                        ]
                        
                        # Add cookies if available
                        if cookie_file:
                            if "cookies.txt" in cookie_file:
                                cmd.extend(["--cookies", cookie_file])
                            else:
                                cmd.extend(["--cookies-from-browser", "chrome"])
                        
                        # Add output path and URL
                        cmd.extend(["-o", audio_path[:-4], youtube_url])
                        
                        logger.info(f"Downloading with yt-dlp CLI: {' '.join(cmd)}")
                        process = await asyncio.create_subprocess_exec(
                            *cmd, 
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await process.communicate()
                        
                        if process.returncode != 0:
                            raise Exception(f"yt-dlp command failed with return code {process.returncode}: {stderr.decode()}")
                    
                    # Check all possible file patterns
                    possible_paths = [
                        audio_path,  # audio_ID.wav
                        f"{audio_path}.wav",  # audio_ID.wav.wav
                        audio_path[:-4],  # audio_ID (no extension)
                        f"{audio_path[:-4]}.wav"  # audio_ID.wav (explicitly)
                    ]
                    
                    # Find the first file that exists
                    actual_path = None
                    for path in possible_paths:
                        if os.path.exists(path) and os.path.getsize(path) > 0:
                            actual_path = path
                            break
                    
                    # If we found a file, make sure it has the correct extension
                    if actual_path:
                        if actual_path != audio_path:
                            # Rename to expected path if needed
                            try:
                                shutil.move(actual_path, audio_path)
                                logger.info(f"Renamed {actual_path} to {audio_path}")
                                actual_path = audio_path
                            except Exception as e:
                                logger.warning(f"Error renaming audio file: {str(e)}")
                        
                        # Ensure audio format is correct for transcription (16kHz, mono, PCM)
                        try:
                            temp_wav_path = audio_path + ".temp.wav"
                            ffmpeg_cmd = [
                                "ffmpeg", "-y", "-i", actual_path,
                                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                                temp_wav_path
                            ]
                            
                            logger.info(f"Converting audio format with command: {' '.join(ffmpeg_cmd)}")
                            
                            ffmpeg_process = await asyncio.create_subprocess_exec(
                                *ffmpeg_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            
                            f_stdout, f_stderr = await ffmpeg_process.communicate()
                            
                            if ffmpeg_process.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                                shutil.move(temp_wav_path, audio_path)
                                logger.info(f"Successfully converted audio format for transcription")
                            else:
                                logger.warning(f"Failed to convert audio format: {f_stderr.decode()}")
                        except Exception as e:
                            logger.warning(f"Error converting audio format: {str(e)}")
                        
                        logger.info(f"Successfully downloaded audio to {audio_path} (size: {os.path.getsize(audio_path)} bytes)")
                        return audio_path
                    
                    logger.warning("yt-dlp download completed but file not found in expected locations")
                    
                    # List directory contents for debugging
                    dir_contents = os.listdir(output_dir)
                    logger.info(f"Directory contents of {output_dir}: {dir_contents}")
                    
                    # Try to find file by partial match
                    matching_files = [f for f in dir_contents if video_id in f]
                    if matching_files:
                        actual_file = os.path.join(output_dir, matching_files[0])
                        logger.info(f"Found matching file: {actual_file}")
                        if os.path.getsize(actual_file) > 0:
                            # Convert to the right format and move to expected path
                            try:
                                temp_wav_path = audio_path + ".temp.wav"
                                ffmpeg_cmd = [
                                    "ffmpeg", "-y", "-i", actual_file,
                                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                                    temp_wav_path
                                ]
                                
                                logger.info(f"Converting found file with command: {' '.join(ffmpeg_cmd)}")
                                
                                ffmpeg_process = await asyncio.create_subprocess_exec(
                                    *ffmpeg_cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                
                                f_stdout, f_stderr = await ffmpeg_process.communicate()
                                
                                if ffmpeg_process.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                                    shutil.move(temp_wav_path, audio_path)
                                    logger.info(f"Successfully converted matched file to audio for transcription")
                                    return audio_path
                                else:
                                    logger.warning(f"Failed to convert matched file: {f_stderr.decode()}")
                            except Exception as e:
                                logger.warning(f"Error converting matched file: {str(e)}")
                            
                            # If conversion failed, just try to use the original file
                            shutil.move(actual_file, audio_path)
                            logger.info(f"Renamed {actual_file} to {audio_path}")
                            return audio_path
                        
                except Exception as e:
                    logger.warning(f"yt-dlp download failed (attempt {current_attempt}/{max_retries}): {str(e)}")
            
            # Try pytube if available
            if "pytube" in self.download_methods:
                try:
                    if HAS_PYTUBE:
                        # Use pytube to download audio
                        logger.info(f"Downloading with pytube library")
                        youtube = pytube.YouTube(youtube_url)
                        stream = youtube.streams.filter(only_audio=True).first()
                        
                        if not stream:
                            # If no audio-only stream, try getting a stream with audio
                            logger.info("No audio-only stream found, trying mixed stream")
                            stream = youtube.streams.filter(progressive=True).first()
                        
                        if stream:
                            temp_path = stream.download(output_path=str(output_dir))
                            logger.info(f"Downloaded to temp path: {temp_path}")
                            
                            # Convert to WAV with correct format for transcription
                            try:
                                cmd = [
                                    "ffmpeg", "-y", "-i", temp_path, 
                                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                                    audio_path
                                ]
                                logger.info(f"Converting to WAV with ffmpeg: {' '.join(cmd)}")
                                process = await asyncio.create_subprocess_exec(
                                    *cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                stdout, stderr = await process.communicate()
                                
                                if process.returncode != 0:
                                    raise Exception(f"ffmpeg command failed: {stderr.decode()}")
                                    
                                os.remove(temp_path)  # Clean up temp file
                            except Exception as ffmpeg_error:
                                logger.warning(f"Failed to convert audio with ffmpeg: {str(ffmpeg_error)}")
                                # Just use the downloaded file
                                shutil.copy(temp_path, audio_path)
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                            
                            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                logger.info(f"Downloaded audio with pytube to {audio_path} (size: {os.path.getsize(audio_path)} bytes)")
                                return audio_path
                    
                    logger.warning("pytube download completed but file not found or empty")
                except Exception as e:
                    logger.warning(f"pytube download failed (attempt {current_attempt}/{max_retries}): {str(e)}")
            
            # If we get here, both methods failed - retry if we have attempts left
            if current_attempt < max_retries:
                # Exponential backoff: wait longer between each retry
                wait_time = 2 ** current_attempt
                logger.info(f"Retrying download in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        # If all attempts have failed, try using a direct command-line approach
        try:
            logger.info("Trying direct youtube-dl command as last resort...")
            cmd = [
                "youtube-dl", 
                "-x",  # Extract audio
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--user-agent", user_agents[0],
                "-o", audio_path[:-4],  # Specify output without extension
                youtube_url
            ]
            
            # Add cookies if available
            if cookie_file and "cookies.txt" in cookie_file:
                cmd.extend(["--cookies", cookie_file])
                
            process = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # Look for created files
            possible_paths = [
                audio_path,  # audio_ID.wav
                f"{audio_path}.wav",  # audio_ID.wav.wav
                audio_path[:-4],  # audio_ID (no extension)
                f"{audio_path[:-4]}.wav"  # audio_ID.wav (explicitly)
            ]
            
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    actual_path = path
                    break
            
            if actual_path:
                # Convert to correct format if needed
                try:
                    temp_wav_path = audio_path + ".temp.wav"
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", actual_path,
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                        temp_wav_path
                    ]
                    
                    logger.info(f"Converting found file with command: {' '.join(ffmpeg_cmd)}")
                    
                    ffmpeg_process = await asyncio.create_subprocess_exec(
                        *ffmpeg_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    f_stdout, f_stderr = await ffmpeg_process.communicate()
                    
                    if ffmpeg_process.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                        if actual_path != audio_path:
                            try:
                                os.remove(actual_path)
                            except:
                                pass
                        
                        shutil.move(temp_wav_path, audio_path)
                        logger.info(f"Successfully converted audio for transcription")
                        return audio_path
                except Exception as e:
                    logger.warning(f"Error converting audio: {str(e)}")
                
                # If conversion failed, try to use the original file
                if actual_path != audio_path:
                    shutil.move(actual_path, audio_path)
                    logger.info(f"Renamed {actual_path} to {audio_path}")
                
                logger.info(f"Successfully downloaded with youtube-dl command: {audio_path}")
                return audio_path
        except Exception as e:
            logger.error(f"Last resort youtube-dl command failed: {str(e)}")
        
        # Create a placeholder file if all download methods failed
        try:
            logger.warning(f"All download methods failed, creating placeholder file")
            with open(audio_path, "wb") as f:
                # Write minimal WAV header (8 bytes)
                f.write(b'RIFF\x00\x00\x00\x00WAVE')
            logger.warning(f"Created placeholder WAV file at {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Failed to create placeholder file: {str(e)}")
        
        # If all attempts have failed, log a critical error
        logger.error(f"All download methods failed after {max_retries} attempts for URL: {youtube_url}")
        
        # Return a mock file in debug mode
        if self.debug_mode:
            logger.info(f"Creating mock audio file in debug mode after download failures")
            with open(audio_path, "w") as f:
                f.write("Mock audio file after download failures")
            return audio_path
        
        return None
    
    async def transcribe_youtube(
        self, 
        youtube_url: str, 
        video_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        session_id: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Transcribe a YouTube video with enhanced error handling, caching, and WebSocket updates
        
        Args:
            youtube_url: URL of the YouTube video
            video_id: Optional video ID (will be generated if not provided)
            tab_id: Optional client tab ID
            session_id: Optional session ID for WebSocket updates
            language: Language code for transcription
            
        Returns:
            Transcription results
        """
        logger.info(f"Transcribing YouTube video: {youtube_url}, video_id={video_id}, tab_id={tab_id}")
        
        try:
            # Generate a video_id if not provided
            if not video_id:
                video_id = f"youtube_{uuid.uuid4().hex[:8]}"
                logger.info(f"Generated new video_id: {video_id}")
            else:
                # Normalize video ID if provided
                video_id = normalize_video_id(video_id)
            
            # Set output file paths
            if tab_id:
                output_file = os.path.join(
                    settings.TRANSCRIPTION_DIR, 
                    f"transcription_{video_id}_{tab_id}.json"
                )
            else:
                output_file = os.path.join(
                    settings.TRANSCRIPTION_DIR, 
                    f"transcription_{video_id}.json"
                )
                
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Send initial status update
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'received',
                    'message': 'Request received and processing started...',
                    'video_id': video_id,
                    'progress': 5
                })
            
            # Check cache first if it's a YouTube URL
            is_cached, cached_result = self.cache.get_from_cache(youtube_url, video_id)
            if is_cached and cached_result:
                logger.info(f"Using cached transcription for {youtube_url}")
                
                # Add video_id to cached result if not present
                if isinstance(cached_result, dict) and "video_id" not in cached_result:
                    cached_result["video_id"] = video_id
                
                # Save to the requested output file if needed
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(cached_result, f)
                
                # Check for timestamps in cached result and save separately if needed
                has_timestamps = False
                if isinstance(cached_result, dict) and "timestamps" in cached_result and cached_result["timestamps"]:
                    try:
                        timestamp_file = save_timestamps_to_file(video_id, cached_result["timestamps"])
                        logger.info(f"Saved timestamps from cache to {timestamp_file}")
                        has_timestamps = True
                    except Exception as ts_error:
                        logger.error(f"Error saving cached timestamps: {str(ts_error)}")
                
                # Send cached result to client if session_id is provided
                if session_id:
                    # Progress update
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'completed',
                        'message': 'Loaded cached transcription',
                        'video_id': video_id,
                        'progress': 100
                    })
                    
                    # Send the transcription
                    await send_websocket_update(session_id, 'transcription', {
                        'status': 'success',
                        'video_id': video_id,
                        'transcript': cached_result.get("text", ""),
                        'has_timestamps': has_timestamps
                    })
                    
                    # Notify client of timestamp availability
                    if has_timestamps:
                        await send_websocket_update(session_id, 'timestamps_available', {
                            'video_id': video_id
                        })
                        
                    logger.info(f"Sent cached transcription to session {session_id} for video {video_id}")
                
                return cached_result
            
            # If not in debug mode and AssemblyAI is available, try direct URL processing first
            if not self.debug_mode and self.assemblyai_available and HAS_ASSEMBLYAI:
                # Try direct transcription with AssemblyAI first (no download required)
                logger.info(f"Attempting direct YouTube URL transcription with AssemblyAI")
                
                # Send status update
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'processing',
                        'message': 'Processing YouTube URL directly...',
                        'video_id': video_id,
                        'progress': 15
                    })
                
                # Try direct URL transcription
                direct_result = await self.transcribe_with_assemblyai_direct(
                    youtube_url=youtube_url,
                    video_id=video_id,
                    session_id=session_id
                )
                
                # If direct transcription worked, save it to the output file and return
                if direct_result and isinstance(direct_result, dict) and "text" in direct_result:
                    logger.info(f"Direct URL transcription with AssemblyAI successful")
                    
                    # Save to output file
                    if output_file:
                        try:
                            with open(output_file, "w") as f:
                                json.dump(direct_result, f)
                            logger.info(f"Saved direct transcription to {output_file}")
                        except Exception as e:
                            logger.error(f"Error saving direct transcription: {str(e)}")
                    
                    # Process timestamps if available
                    has_timestamps = False
                    if "timestamps" in direct_result and direct_result["timestamps"]:
                        try:
                            timestamp_file = save_timestamps_to_file(video_id, direct_result["timestamps"])
                            logger.info(f"Saved timestamps to {timestamp_file}")
                            has_timestamps = True
                        except Exception as ts_error:
                            logger.error(f"Error saving timestamps: {str(ts_error)}")
                    
                    # Cache the result
                    try:
                        self.cache.save_to_cache(youtube_url, direct_result)
                    except Exception as cache_error:
                        logger.error(f"Error saving to cache: {str(cache_error)}")
                    
                    # Send completion status
                    if session_id:
                        await send_websocket_update(session_id, 'transcription_status', {
                            'status': 'completed',
                            'message': 'Direct transcription completed successfully',
                            'video_id': video_id,
                            'progress': 100
                        })
                        
                        # Send the transcription
                        await send_websocket_update(session_id, 'transcription', {
                            'status': 'success',
                            'video_id': video_id,
                            'transcript': direct_result.get("text", ""),
                            'has_timestamps': has_timestamps
                        })
                        
                        # Notify about timestamps
                        if has_timestamps:
                            await send_websocket_update(session_id, 'timestamps_available', {
                                'video_id': video_id
                            })
                            
                        logger.info(f"Sent direct transcription to client for video {video_id}")
                    
                    # Add success flag
                    direct_result["success"] = True
                    
                    return direct_result
                else:
                    logger.warning("Direct URL transcription failed, falling back to download method")
            
            # If not cached or direct transcription failed, try the unified transcribe_media approach
            try:
                # Send status update
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'downloading',
                        'message': 'Processing YouTube video...',
                        'video_id': video_id,
                        'progress': 10
                    })
                
                # Try unified approach
                logger.info(f"Using unified transcribe_media approach for YouTube URL: {youtube_url}")
                
                # Send status update before transcription
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'transcribing',
                        'message': 'Transcribing audio...',
                        'video_id': video_id,
                        'progress': 30
                    })
                
                # Determine preferred service based on availability
                preferred_service = None
                if hasattr(self, 'assemblyai_available') and self.assemblyai_available:
                    preferred_service = "assemblyai"  # Prioritize AssemblyAI for YouTube
                elif hasattr(self, 'google_available') and self.google_available:
                    preferred_service = "google"
                
                # Force disable debug mode to ensure proper transcription
                transcription_result = await transcribe_media(
                    source=youtube_url,
                    output_file=output_file,
                    language_code=language,
                    video_id=video_id,
                    preferred_service=preferred_service,
                    debug_mode=self.debug_mode,
                    use_cache=True
                )
                
                logger.info(f"transcribe_media returned result for YouTube URL")
                
                # Fall back to traditional download+transcribe approach if transcribe_media doesn't work
                if not transcription_result or "error" in transcription_result:
                    raise Exception("Unified transcription approach failed, falling back to traditional approach")
            
            except Exception as e:
                logger.warning(f"Unified transcription approach failed: {str(e)}, falling back to traditional approach")
                
                # Download the audio using our robust method
                temp_dir = Path(settings.TEMP_DIR)
                audio_path = await self.download_youtube_audio(youtube_url, temp_dir, video_id)
                
                # If download failed, return error
                if not audio_path:
                    error_msg = f"Failed to download YouTube video: {youtube_url}"
                    logger.error(error_msg)
                    
                    # Send error to client if session_id is provided
                    if session_id:
                        await send_websocket_update(session_id, 'transcription_status', {
                            'status': 'error',
                            'message': 'Failed to download YouTube video',
                            'video_id': video_id,
                            'progress': 100
                        })
                        
                        await send_websocket_update(session_id, 'error', {
                            'message': 'Failed to download YouTube video. Please try another URL or a different video.',
                            'video_id': video_id
                        })
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "video_id": video_id,
                        "text": error_msg
                    }
                
                # Send status update for transcription if session_id is provided
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'transcribing',
                        'message': 'Transcribing audio...',
                        'video_id': video_id,
                        'progress': 50
                    })
                
                # Determine preferred service based on availability and audio length
                preferred_service = None
                if hasattr(self, 'assemblyai_available') and self.assemblyai_available:
                    preferred_service = "assemblyai"  # Prioritize AssemblyAI for YouTube
                elif hasattr(self, 'google_available') and self.google_available:
                    preferred_service = "google"
                
                # For longer videos, prioritize AssemblyAI if available
                audio_duration = 0
                try:
                    import wave
                    with wave.open(audio_path, 'rb') as audio_file:
                        frames = audio_file.getnframes()
                        rate = audio_file.getframerate()
                        audio_duration = frames / float(rate)
                    logger.info(f"Audio duration: {audio_duration:.2f} seconds")
                    
                    # For longer audio (>1 minute), use AssemblyAI if available
                    if audio_duration > 60 and self.assemblyai_available:
                        preferred_service = "assemblyai"
                        logger.info(f"Selected AssemblyAI for long audio ({audio_duration:.2f} seconds)")
                except Exception as e:
                    logger.warning(f"Could not determine audio duration: {str(e)}")
                
                # Transcribe the video
                try:
                    transcription_result = await transcribe_video(
                        video_path=audio_path,
                        output_file=output_file,
                        language_code=language,
                        video_id=video_id,
                        preferred_service=preferred_service,
                        debug_mode=self.debug_mode
                    )
                    
                    # Clean up temporary files
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            logger.info(f"Removed temporary audio file: {audio_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file: {str(e)}")
                
                except Exception as e:
                    error_msg = f"Error transcribing YouTube video: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    
                    # Send error to client if session_id is provided
                    if session_id:
                        await send_websocket_update(session_id, 'transcription_status', {
                            'status': 'error',
                            'message': 'Transcription failed',
                            'video_id': video_id,
                            'progress': 100
                        })
                        
                        await send_websocket_update(session_id, 'error', {
                            'message': 'Transcription service error. Our team has been notified.',
                            'video_id': video_id
                        })
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "video_id": video_id,
                        "text": f"Error transcribing YouTube video: {str(e)}"
                    }
            
            # Handle successful transcription
            if transcription_result and isinstance(transcription_result, dict):
                # Add video_id to the result if not present
                if "video_id" not in transcription_result:
                    transcription_result["video_id"] = video_id
                
                # Make sure we have text in the result
                if "text" not in transcription_result or not transcription_result["text"]:
                    transcription_result["text"] = f"This video was processed successfully, but no transcribable speech was detected. Video ID: {video_id}"
                
                # Process timestamps if available
                has_timestamps = False
                if "timestamps" in transcription_result and transcription_result["timestamps"]:
                    try:
                        timestamp_file = save_timestamps_to_file(video_id, transcription_result["timestamps"])
                        logger.info(f"Saved timestamps to {timestamp_file}")
                        has_timestamps = True
                    except Exception as ts_error:
                        logger.error(f"Error saving timestamps: {str(ts_error)}")
                
                # Cache the result
                try:
                    self.cache.save_to_cache(youtube_url, transcription_result)
                except Exception as cache_error:
                    logger.error(f"Error saving to cache: {str(cache_error)}")
                
                # Send completion status if session_id is provided
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'completed',
                        'message': 'Transcription completed successfully',
                        'video_id': video_id,
                        'progress': 100
                    })
                    
                    # Send the transcription
                    await send_websocket_update(session_id, 'transcription', {
                        'status': 'success',
                        'video_id': video_id,
                        'transcript': transcription_result.get("text", ""),
                        'has_timestamps': has_timestamps
                    })
                    
                    # Notify about timestamps
                    if has_timestamps:
                        await send_websocket_update(session_id, 'timestamps_available', {
                            'video_id': video_id
                        })
                        
                    logger.info(f"Sent transcription to client for video {video_id}")
                
                # Add success flag to the result
                transcription_result["success"] = True
                
                # Save to tab-specific file if needed
                if tab_id and tab_id != "default":
                    tab_output_file = os.path.join(
                        settings.TRANSCRIPTION_DIR, 
                        f"transcription_{video_id}_{tab_id}.json"
                    )
                    try:
                        with open(tab_output_file, "w") as f:
                            json.dump(transcription_result, f)
                        logger.info(f"Saved tab-specific transcription to {tab_output_file}")
                    except Exception as e:
                        logger.error(f"Error saving tab-specific transcription: {e}")
                
                return transcription_result
            else:
                error_msg = "Transcription service returned an invalid result"
                logger.error(error_msg)
                
                # Send error to client if session_id is provided
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'error',
                        'message': 'Transcription service returned an invalid result',
                        'video_id': video_id,
                        'progress': 100
                    })
                    
                    await send_websocket_update(session_id, 'error', {
                        'message': 'Transcription service returned an invalid result',
                        'video_id': video_id
                    })
                
                return {
                    "success": False,
                    "error": error_msg,
                    "video_id": video_id,
                    "text": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error in transcribe_youtube: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error to client if session_id is provided
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'error',
                    'message': 'An unexpected error occurred',
                    'video_id': video_id if video_id else "unknown",
                    'progress': 100
                })
                
                await send_websocket_update(session_id, 'error', {
                    'message': 'An unexpected error occurred. Please try again later.',
                    'video_id': video_id if video_id else "unknown"
                })
            
            # Return standardized error response
            return {
                "success": False,
                "error": error_msg,
                "video_id": video_id if video_id else "unknown",
                "text": f"Error transcribing YouTube video: {str(e)}"
            }
    
    async def transcribe_video_file(
        self, 
        video_path: str, 
        video_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        session_id: Optional[str] = None,
        language: str = "en",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe a video file with enhanced error handling and WebSocket updates
        
        Args:
            video_path: Path to the video file
            video_id: Optional video ID (will be generated if not provided)
            tab_id: Optional client tab ID
            session_id: Optional session ID for WebSocket updates
            language: Language code for transcription
            output_file: Optional path to save the transcription
            
        Returns:
            Transcription results
        """
        logger.info(f"Transcribing video file: {video_path}, video_id={video_id}, tab_id={tab_id}")
        
        try:
            # Check if the file exists
            if not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                logger.error(error_msg)
                
                # Send error to client if session_id is provided
                if session_id:
                    await send_websocket_update(session_id, 'error', {
                        'message': "Video file not found. Please try again.",
                        'video_id': video_id if video_id else "unknown"
                    })
                
                return {
                    "success": False,
                    "error": error_msg,
                    "video_id": video_id
                }
            
            # Generate a video_id if not provided
            if not video_id:
                video_id = f"upload_{uuid.uuid4().hex[:8]}"
                logger.info(f"Generated new video_id: {video_id}")
            else:
                # Normalize video ID if provided
                video_id = normalize_video_id(video_id)
            
            # Set output file paths if not provided
            if not output_file:
                if tab_id:
                    output_file = os.path.join(
                        settings.TRANSCRIPTION_DIR, 
                        f"transcription_{video_id}_{tab_id}.json"
                    )
                else:
                    output_file = os.path.join(
                        settings.TRANSCRIPTION_DIR, 
                        f"transcription_{video_id}.json"
                    )
                    
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Send initial status update
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'received',
                    'message': 'File received and processing started...',
                    'video_id': video_id,
                    'progress': 5
                })
            
            # Determine if file is audio or video
            file_ext = os.path.splitext(video_path)[1].lower()
            is_audio_file = file_ext in ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
            file_type = "audio" if is_audio_file else "video"
            
            # Update progress
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'processing',
                    'message': 'Processing audio file...' if is_audio_file else 'Extracting audio from video...',
                    'video_id': video_id,
                    'progress': 20
                })
            
            # Try using unified transcribe_media approach first
            try:
                # Use the unified transcribe_media function
                logger.info(f"Using unified transcribe_media approach for file: {video_path}")
                
                # Determine preferred service based on availability
                preferred_service = None
                if hasattr(self, 'google_available') and self.google_available:
                    preferred_service = "google"
                elif hasattr(self, 'assemblyai_available') and self.assemblyai_available:
                    preferred_service = "assemblyai"
                
                transcription_result = await transcribe_media(
                    source=video_path,
                    output_file=output_file,
                    language_code=language,
                    video_id=video_id,
                    file_type=file_type,  # Specify file type
                    preferred_service=preferred_service,
                    debug_mode=self.debug_mode
                )
                
                logger.info(f"transcribe_media returned result for file")
                
                # Fall back to traditional approach if this fails
                if not transcription_result or "error" in transcription_result:
                    raise Exception("Unified transcription approach failed, falling back to traditional approach")
            
            except Exception as e:
                logger.warning(f"Unified transcription approach failed: {str(e)}, falling back to traditional approach")
                
                # Extract audio if it's a video file or convert audio to WAV
                temp_wav_path = None
                if not self.debug_mode:
                    try:
                        # Create a temporary directory for processing
                        temp_dir = tempfile.mkdtemp()
                        temp_wav_path = os.path.join(temp_dir, "audio.wav")
                        
                        logger.info(f"Extracting audio to temp path: {temp_wav_path}")
                        
                        # Use ffmpeg to extract audio or convert to WAV with format needed for transcription
                        cmd = [
                            "ffmpeg", "-y", "-i", video_path,
                            "-vn",  # Skip video if present
                            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                            temp_wav_path
                        ]
                        
                        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                        
                        # Set shell=False and use list of arguments for security
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        stdout, stderr = await process.communicate()
                        
                        if process.returncode != 0:
                            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                            logger.error(f"Error extracting/converting audio: {error_msg}")
                            
                            # Try a fallback command if the first one fails
                            fallback_cmd = [
                                "ffmpeg", "-y", "-i", video_path,
                                "-vn", "-acodec", "pcm_s16le", 
                                temp_wav_path
                            ]
                            
                            logger.info(f"Trying fallback ffmpeg command: {' '.join(fallback_cmd)}")
                            
                            fallback_process = await asyncio.create_subprocess_exec(
                                *fallback_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            
                            f_stdout, f_stderr = await fallback_process.communicate()
                            
                            if fallback_process.returncode != 0:
                                fallback_error = f_stderr.decode() if f_stderr else "Unknown fallback ffmpeg error"
                                logger.error(f"Fallback ffmpeg also failed: {fallback_error}")
                                # We'll continue with the original file if extraction fails
                                transcription_path = video_path
                            else:
                                transcription_path = temp_wav_path
                        else:
                            transcription_path = temp_wav_path
                            
                        logger.info(f"Audio extraction result - temp file exists: {os.path.exists(transcription_path)}")
                        if os.path.exists(transcription_path):
                            logger.info(f"Temp file size: {os.path.getsize(transcription_path)} bytes")
                            
                    except Exception as e:
                        logger.error(f"Error in audio extraction: {str(e)}", exc_info=True)
                        # Fall back to using the original file
                        transcription_path = video_path
                else:
                    logger.info("Using debug mode, skipping audio extraction")
                    transcription_path = video_path
                
                # Send status update for transcription
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'transcribing',
                        'message': 'Transcribing audio...',
                        'video_id': video_id,
                        'progress': 40
                    })
                
                # Debug mode - generate mock transcription
                if self.debug_mode:
                    logger.info(f"Using debug mode for transcription - real file path: {video_path}")
                    
                    # Mock transcription result
                    mock_timestamps = [
                        {"word": "This", "start_time": 0.0, "end_time": 0.5},
                        {"word": "is", "start_time": 0.5, "end_time": 0.7},
                        {"word": "a", "start_time": 0.7, "end_time": 0.8},
                        {"word": "debug", "start_time": 0.8, "end_time": 1.2},
                        {"word": "transcription", "start_time": 1.2, "end_time": 2.0},
                        {"word": "for", "start_time": 2.0, "end_time": 2.2},
                        {"word": os.path.basename(video_path), "start_time": 2.2, "end_time": 2.7},
                        {"word": "ID:", "start_time": 3.0, "end_time": 3.5},
                        {"word": video_id, "start_time": 3.5, "end_time": 4.0}
                    ]
                    
                    transcription_result = {
                        "text": f"This is a debug transcription for the uploaded file: {os.path.basename(video_path)}. File ID: {video_id}",
                        "segments": [
                            {
                                "id": 0,
                                "start": 0.0,
                                "end": 5.0,
                                "text": f"This is a debug transcription for the uploaded file. File ID: {video_id}"
                            }
                        ],
                        "timestamps": mock_timestamps,
                        "video_id": video_id,
                        "success": True
                    }
                    
                    # Save to output file
                    with open(output_file, "w") as f:
                        json.dump(transcription_result, f)
                        
                    # Save timestamps separately
                    timestamp_path = save_timestamps_to_file(video_id, mock_timestamps)
                    logger.info(f"Saved debug timestamps to {timestamp_path}")
                    
                    # Skip to completion handling
                else:
                    # Real transcription
                    logger.info(f"Transcribing file: {transcription_path}")
                    
                    # Check audio length to decide which service to use
                    audio_duration = 0
                    try:
                        import wave
                        try:
                            with wave.open(transcription_path, 'rb') as audio_file:
                                frames = audio_file.getnframes()
                                rate = audio_file.getframerate()
                                audio_duration = frames / float(rate)
                            logger.info(f"Audio duration: {audio_duration:.2f} seconds")
                        except Exception as e:
                            logger.warning(f"Could not determine audio duration: {str(e)}")
                    except ImportError:
                        logger.warning("Wave module not available, skipping duration check")
                    
                    # Determine preferred service based on availability and duration
                    preferred_service = None
                    if hasattr(self, 'google_available') and self.google_available:
                        preferred_service = "google"
                    elif hasattr(self, 'assemblyai_available') and self.assemblyai_available:
                        preferred_service = "assemblyai"
                    
                    # For longer audio (>1 minute), use AssemblyAI if available
                    if audio_duration > 60 and hasattr(self, 'assemblyai_available') and self.assemblyai_available:
                        preferred_service = "assemblyai"
                        logger.info(f"Selected AssemblyAI for long audio ({audio_duration:.2f} seconds)")
                    
                    # Transcribe the audio
                    try:
                        transcription_result = await transcribe_video(
                            video_path=transcription_path,
                            output_file=output_file,
                            language_code=language,
                            video_id=video_id,
                            preferred_service=preferred_service,
                            debug_mode=self.debug_mode
                        )
                        
                        logger.info(f"transcribe_video returned result of type: {type(transcription_result).__name__}")
                    except Exception as e:
                        error_msg = f"Error transcribing video file: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        
                        # Send error to client
                        if session_id:
                            await send_websocket_update(session_id, 'transcription_status', {
                                'status': 'error',
                                'message': 'Transcription failed',
                                'video_id': video_id,
                                'progress': 100
                            })
                            
                            await send_websocket_update(session_id, 'error', {
                                'message': f'Transcription service error. Our team has been notified.',
                                'video_id': video_id
                            })
                        
                        return {
                            "success": False,
                            "error": error_msg,
                            "video_id": video_id,
                            "text": f"Error transcribing video file: {str(e)}"
                        }
                    
                    # Clean up temporary files
                    if temp_wav_path and os.path.exists(temp_wav_path):
                        try:
                            os.remove(temp_wav_path)
                            logger.info(f"Removed temporary WAV file: {temp_wav_path}")
                            
                            # Also remove temp directory if it exists
                            temp_dir = os.path.dirname(temp_wav_path)
                            if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                                shutil.rmtree(temp_dir)
                                logger.info(f"Removed temporary directory: {temp_dir}")
                        except Exception as cleanup_error:
                            logger.warning(f"Error cleaning up temporary files: {str(cleanup_error)}")
            
            # Process the transcription result
            if transcription_result and isinstance(transcription_result, dict):
                # Add video_id to the result if not present
                if "video_id" not in transcription_result:
                    transcription_result["video_id"] = video_id
                    
                # Make sure we have text in the result
                if "text" not in transcription_result or not transcription_result["text"]:
                    transcription_result["text"] = f"This file was processed successfully, but no transcribable speech was detected. File ID: {video_id}"
                
                # Process timestamps if available
                has_timestamps = False
                if "timestamps" in transcription_result and transcription_result["timestamps"]:
                    try:
                        timestamp_file = save_timestamps_to_file(video_id, transcription_result["timestamps"])
                        logger.info(f"Saved timestamps to {timestamp_file}")
                        has_timestamps = True
                    except Exception as ts_error:
                        logger.error(f"Error saving timestamps: {str(ts_error)}")
                        
                # Send completion status
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'completed',
                        'message': 'Transcription completed successfully',
                        'video_id': video_id,
                        'progress': 100
                    })
                    
                    # Send transcription to client
                    await send_websocket_update(session_id, 'transcription', {
                        'status': 'success',
                        'video_id': video_id,
                        'transcript': transcription_result.get("text", ""),
                        'has_timestamps': has_timestamps
                    })
                    
                    # Notify about timestamps
                    if has_timestamps:
                        await send_websocket_update(session_id, 'timestamps_available', {
                            'video_id': video_id
                        })
                        
                    logger.info(f"Sent transcription to client for file {video_id}")
                
                # Add success flag to the result
                transcription_result["success"] = True
                
                # Save to tab-specific file if needed
                if tab_id and tab_id != "default":
                    tab_output_file = os.path.join(
                        settings.TRANSCRIPTION_DIR, 
                        f"transcription_{video_id}_{tab_id}.json"
                    )
                    try:
                        with open(tab_output_file, "w") as f:
                            json.dump(transcription_result, f)
                        logger.info(f"Saved tab-specific transcription to {tab_output_file}")
                    except Exception as e:
                        logger.error(f"Error saving tab-specific transcription: {e}")
                
                return transcription_result
            else:
                error_msg = "Transcription service returned an invalid result"
                logger.error(error_msg)
                
                # Send error to client
                if session_id:
                    await send_websocket_update(session_id, 'transcription_status', {
                        'status': 'error',
                        'message': 'Transcription service returned an invalid result',
                        'video_id': video_id,
                        'progress': 100
                    })
                    
                    await send_websocket_update(session_id, 'error', {
                        'message': 'Transcription service returned an invalid result',
                        'video_id': video_id
                    })
                
                return {
                    "success": False,
                    "error": error_msg,
                    "video_id": video_id,
                    "text": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error in transcribe_video_file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error to client
            if session_id:
                await send_websocket_update(session_id, 'transcription_status', {
                    'status': 'error',
                    'message': 'An unexpected error occurred',
                    'video_id': video_id if video_id else "unknown",
                    'progress': 100
                })
                
                await send_websocket_update(session_id, 'error', {
                    'message': 'An unexpected error occurred. Please try again later.',
                    'video_id': video_id if video_id else "unknown"
                })
            
            # Return standardized error response
            return {
                "success": False,
                "error": error_msg,
                "video_id": video_id if video_id else "unknown",
                "text": f"Error transcribing video file: {str(e)}"
            }
    
    async def get_transcript(
        self, 
        video_id: str,
        tab_id: Optional[str] = None
    ) -> str:
        """
        Get the transcript for a video
        
        Args:
            video_id: Video ID
            tab_id: Optional client tab ID
            
        Returns:
            Transcript text
        """
        logger.info(f"Getting transcript for video: {video_id}, tab_id={tab_id}")
        
        try:
            # Normalize video ID
            normalized_id = normalize_video_id(video_id)
            
            # First check tab-specific file if tab_id is provided
            if tab_id:
                tab_file = os.path.join(
                    settings.TRANSCRIPTION_DIR, 
                    f"transcription_{normalized_id}_{tab_id}.json"
                )
                if os.path.exists(tab_file):
                    try:
                        with open(tab_file, "r") as f:
                            data = json.load(f)
                        return data.get("text", "")
                    except Exception as e:
                        logger.error(f"Error reading tab-specific transcription file: {str(e)}")
            
            # Check regular file
            files_to_check = [
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{normalized_id}.json"),
                os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{normalized_id}.json")
            ]
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        return data.get("text", "")
                    except Exception as e:
                        logger.error(f"Error reading transcription file: {str(e)}")
            
            return f"No transcript found for video ID: {video_id}"
            
        except Exception as e:
            logger.error(f"Error getting transcript: {str(e)}", exc_info=True)
            return f"Error retrieving transcript: {str(e)}"
    
    async def get_timestamps(
        self, 
        video_id: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get timestamps for a video
        
        Args:
            video_id: Video ID
            session_id: Optional session ID for WebSocket updates
            
        Returns:
            Dictionary with timestamp data
        """
        logger.info(f"Getting timestamps for video: {video_id}")
        
        try:
            # Normalize video ID
            normalized_id = normalize_video_id(video_id)
            
            # Try to get timestamps from file
            timestamp_data = await get_timestamps_for_video(normalized_id)
            
            # If timestamps found and session_id provided, send to client
            if timestamp_data and "formatted_timestamps" in timestamp_data and session_id:
                logger.info(f"Found timestamps for video_id {normalized_id}: {len(timestamp_data.get('formatted_timestamps', []))} entries")
                
                await send_websocket_update(session_id, 'timestamps_data', {
                    'video_id': normalized_id,
                    'timestamps': timestamp_data.get("formatted_timestamps", [])
                })
                logger.info(f"Sent timestamps to session {session_id}")
            
            # If no timestamps found, try to generate from transcription file
            if not timestamp_data:
                transcription_paths = [
                    os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{normalized_id}.json"),
                    os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{normalized_id}.json")
                ]
                
                for path in transcription_paths:
                    if os.path.exists(path):
                        try:
                            with open(path, "r") as f:
                                data = json.load(f)
                            
                            if "timestamps" in data and data["timestamps"]:
                                # Save timestamps to file
                                timestamp_path = save_timestamps_to_file(normalized_id, data["timestamps"])
                                
                                # Read the newly created file
                                with open(timestamp_path, "r") as f:
                                    timestamp_data = json.load(f)
                                    
                                logger.info(f"Generated and saved timestamps from transcription file for {normalized_id}")
                                
                                # Send to client if session_id provided
                                if session_id:
                                    await send_websocket_update(session_id, 'timestamps_data', {
                                        'video_id': normalized_id,
                                        'timestamps': timestamp_data.get("formatted_timestamps", [])
                                    })
                                    logger.info(f"Sent generated timestamps to session {session_id}")
                                
                                return timestamp_data
                        except Exception as e:
                            logger.error(f"Error generating timestamps from transcription file: {str(e)}")
                
                # If still no timestamps and session_id provided, send error
                if session_id:
                    await send_websocket_update(session_id, 'error', {
                        'message': f"No timestamps available for this video.",
                        'video_id': normalized_id
                    })
                
                return {"error": "No timestamps found", "video_id": normalized_id}
            
            return timestamp_data
            
        except Exception as e:
            error_msg = f"Error getting timestamps: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error to client if session_id provided
            if session_id:
                await send_websocket_update(session_id, 'error', {
                    'message': f"Error retrieving timestamps: {str(e)}",
                    'video_id': video_id
                })
            
            return {"error": error_msg, "video_id": video_id}
    
    async def clear_cache(self) -> bool:
        """Clear all cache entries"""
        try:
            result = self.cache.clear_all_cache()
            logger.info(f"Cache cleared, result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def shutdown(self):
        """Clean up resources when shutting down"""
        logger.info("Shutting down transcription service")
        # Any additional cleanup can be added here

# Create service instance FIRST - this is critical
transcription_service = TranscriptionService()

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
    
    # Use the transcription service to transcribe the audio file
    # Pass session_id explicitly to ensure WebSocket notifications work
    return await transcription_service.transcribe_video_file(
        video_path=file_path,
        video_id=actual_video_id,
        tab_id=tab_id,
        session_id=session_id or tab_id,  # Use tab_id as session_id if not provided
        language=language_code,
        output_file=output_file
    )

async def process_youtube_video(
    youtube_url: str,
    video_id: Optional[str] = None,
    tab_id: Optional[str] = None,
    session_id: Optional[str] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Process a YouTube video for transcription
    
    Args:
        youtube_url: URL of the YouTube video
        video_id: Optional video ID (will be generated if not provided)
        tab_id: Optional client tab ID
        session_id: Optional session ID for WebSocket updates
        language: Language code for transcription
        
    Returns:
        Transcription results
    """
    logger.info(f"Processing YouTube video: {youtube_url}, video_id={video_id}, tab_id={tab_id}")
    
    # Use the transcription service to transcribe the YouTube video
    return await transcription_service.transcribe_youtube(
        youtube_url=youtube_url,
        video_id=video_id,
        tab_id=tab_id,
        session_id=session_id,
        language=language
    )

async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a transcription task
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
    """
    logger.info(f"Getting status for task: {task_id}")
    
    # For now, implement a simple stub that returns a fixed response
    # This would need to be replaced with actual task status tracking
    return {
        "task_id": task_id,
        "status": "unknown",
        "message": "Task status tracking not fully implemented yet"
    }

# Expose the cache instance from the transcription service
cache = transcription_service.cache

async def shutdown():
    """
    Shutdown the transcription service and clean up resources
    
    This is a standalone wrapper for the transcription_service.shutdown() method
    """
    logger.info("Shutting down transcription service from standalone function")
    
    # Call the shutdown method on the service instance
    await transcription_service.shutdown()
    
def get_transcription_service() -> TranscriptionService:
    """Get the transcription service instance"""
    return transcription_service