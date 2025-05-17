# app/socketio_server.py
"""
Socket.IO server for Luna AI
Integrated with the main FastAPI application
"""
import socketio
import logging
import traceback
import asyncio
from typing import Dict, Any, Optional
import os
# Import the actual transcription services
from app.services.transcription_service import TranscriptionService
from app.services.video_processing import VideoProcessor
# Import AI service for chat
from app.services.ai import AIService  # Add this import

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("socketio.server")

# Create a Socket.IO server with increased timeout values
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",  # Allow all origins in development
    logger=True,
    engineio_logger=True,
    # Increase timeout values to prevent "Timeout getting..." errors
    ping_timeout=60,     # Increase from default 20 seconds
    ping_interval=25,    # Increase from default 15 seconds
    max_http_buffer_size=5 * 1024 * 1024  # Increase buffer size for larger payloads
)

# Create an ASGI app for Socket.IO
socket_app = socketio.ASGIApp(
    sio,
    socketio_path='',  # Use the default path
    static_files=None
)

# Store session data
session_data = {}

# Store processing tasks to track them
active_tasks = {}

# Store processed messages to avoid duplicates
processed_messages = set()

# Timeout settings for long-running operations
OPERATION_TIMEOUT = 300  # seconds - increased for transcription which can take longer

# Initialize services
transcription_service = TranscriptionService()
video_processor = VideoProcessor()  # Remove the file_id parameter
ai_service = AIService()  # Initialize AI service

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    client_id = environ.get('HTTP_X_CLIENT_ID', sid)
    tab_id = None
    
    # Try to get tabId from query string
    query_string = environ.get('QUERY_STRING', '')
    if 'tabId=' in query_string:
        try:
            tab_id = query_string.split('tabId=')[1].split('&')[0]
        except:
            pass
    
    if tab_id:
        client_id = tab_id
    
    logger.info(f"Client {client_id} connected with SID: {sid}")
    logger.debug(f"Connection details: {environ}")
    
    session_data[sid] = {'client_id': client_id, 'tab_id': tab_id}
    await sio.emit('connection_established', {'status': 'connected', 'sid': sid}, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    logger.info(f"Client {client_id} disconnected. SID: {sid}")
    
    # Cancel any pending tasks for this session
    if sid in active_tasks:
        for task_name, task in active_tasks[sid].items():
            logger.info(f"Cancelling task {task_name} for disconnected client {client_id}")
            try:
                task.cancel()
            except:
                pass
        del active_tasks[sid]
        
    if sid in session_data:
        del session_data[sid]

@sio.event
async def error(sid, error_data):
    """Handle Socket.IO errors"""
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    logger.error(f"Socket.IO error for client {client_id}: {error_data}")

@sio.event
async def register_tab(sid, data):
    """
    Register a tab with the server
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    tab_id = data.get('tab_id')
    logger.info(f"Registering tab with ID: {tab_id} for client {client_id}")
    
    if sid in session_data:
        session_data[sid]['tab_id'] = tab_id
    
    await sio.emit('tab_registered', {
        'status': 'success',
        'tab_id': tab_id
    }, room=sid)

# Add the missing send_message event handler
@sio.event
async def send_message(sid, data):
    """
    Handle chat messages from users and generate AI responses
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    message = data.get('message')
    video_id = data.get('video_id')
    message_id = data.get('messageId')
    
    logger.info(f"Received message from {client_id} for video: {video_id}")
    
    if not message:
        await sio.emit('ai_response_error', {
            'error': 'Missing message',
            'status': 'error'
        }, room=sid)
        return
    
    # Check for duplicate messages to prevent double responses
    message_key = f"{sid}:{message_id or message[:50]}"
    if message_key in processed_messages:
        logger.warning(f"Duplicate message detected: {message_key}")
        return
    
    processed_messages.add(message_key)
    
    # Create a task to handle the AI response with timeout
    async def process_ai_response():
        try:
            # Generate response from AI service
            response = await ai_service.generate_response(message, video_id)
            
            # Send the AI response back to the client
            await sio.emit('ai_response', {
                'message': response,
                'messageId': message_id,
                'status': 'success'
            }, room=sid)
            
        except asyncio.CancelledError:
            logger.info(f"AI response for message '{message[:30]}...' was cancelled")
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}\n{traceback.format_exc()}")
            await sio.emit('ai_response_error', {
                'error': str(e),
                'status': 'error'
            }, room=sid)
    
    try:
        # Create task and register it
        if sid not in active_tasks:
            active_tasks[sid] = {}
        
        task = asyncio.create_task(process_ai_response())
        active_tasks[sid]['ai_response'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=30)  # 30-second timeout for AI responses
        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating AI response for '{message[:30]}...'")
            await sio.emit('ai_response_error', {
                'error': 'Response generation timed out',
                'status': 'timeout'
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'ai_response' in active_tasks[sid]:
                del active_tasks[sid]['ai_response']
            
            # Clean up processed messages after a delay to prevent memory leaks
            # But keep it long enough to catch duplicates in normal operation
            asyncio.create_task(clear_processed_message(message_key, delay=60))
    
    except Exception as e:
        logger.error(f"Error setting up AI response task: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('ai_response_error', {
            'error': str(e),
            'status': 'error'
        }, room=sid)

async def clear_processed_message(message_key, delay=60):
    """Helper to clean up processed messages after a delay"""
    await asyncio.sleep(delay)
    if message_key in processed_messages:
        processed_messages.remove(message_key)

@sio.event
async def echo(sid, data):
    """
    Simple echo event handler for testing Socket.IO connection
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    logger.info(f"Received echo from {client_id}: {data}")
    
    await sio.emit('echo_response', {
        'status': 'success',
        'message': 'Echo received',
        'received_data': data,
        'sid': sid
    }, room=sid)

@sio.event
async def heartbeat(sid, data):
    """
    Handle heartbeat messages to keep the connection alive
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    logger.debug(f"Heartbeat from client {client_id}")
    await sio.emit('heartbeat_response', {'status': 'alive'}, room=sid)

@sio.event
async def transcription(sid, data):
    """
    Handle video transcription requests
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    video_url = data.get('video_url')
    force_refresh = data.get('force_refresh', False)
    
    logger.info(f"Received transcription request from {client_id} for video: {video_id or video_url}")
    
    if not video_id and not video_url:
        await sio.emit('transcription_error', {
            'error': 'Missing video_id or video_url',
            'status': 'error'
        }, room=sid)
        return
    
    # Create an async task to handle transcription with timeout protection
    async def process_transcription():
        try:
            # Update client with initial status
            await sio.emit('transcription_status', {
                'status': 'received',
                'message': 'Request received and processing started...',
                'video_id': video_id,
                'progress': 0
            }, room=sid)
            
            # Check cache first if not forcing refresh
            if not force_refresh and video_id:
                # Update status to checking cache
                await sio.emit('transcription_status', {
                    'status': 'checking_cache',
                    'message': 'Checking for cached transcription...',
                    'video_id': video_id,
                    'progress': 5
                }, room=sid)
                
                # Check for cached transcription
                cached_transcript = await transcription_service.get_cached_transcription(video_id)
                
                if cached_transcript:
                    await sio.emit('transcription_status', {
                        'status': 'found_cache',
                        'message': 'Found cached transcription, loading...',
                        'video_id': video_id,
                        'progress': 50
                    }, room=sid)
                    
                    # Send the cached transcription data
                    await sio.emit('transcription_data', {
                        'status': 'success',
                        'video_id': video_id,
                        'transcript': cached_transcript['text'],
                        'has_timestamps': cached_transcript.get('has_timestamps', False)
                    }, room=sid)
                    
                    # Set transcription length for UI
                    await sio.emit('transcription_length', {
                        'video_id': video_id,
                        'length': len(cached_transcript['text'])
                    }, room=sid)
                    
                    await sio.emit('transcription_status', {
                        'status': 'completed',
                        'message': 'Loaded cached transcription',
                        'video_id': video_id,
                        'progress': 100
                    }, room=sid)
                    
                    return
            
            # If we get here, we need to process the video
            # Update status to downloading
            await sio.emit('transcription_status', {
                'status': 'downloading',
                'message': 'Downloading video...',
                'video_id': video_id,
                'progress': 10
            }, room=sid)
            
            # Download the video if needed
            video_path = None
            if video_url:
                video_path = await video_processor.download_video(video_url)
                if not video_path:
                    raise Exception("Failed to download video")
            
            # Extract audio
            await sio.emit('transcription_status', {
                'status': 'processing',
                'message': 'Extracting audio...',
                'video_id': video_id,
                'progress': 30
            }, room=sid)
            
            audio_path = await video_processor.extract_audio(video_path or video_id)
            if not audio_path:
                raise Exception("Failed to extract audio")
            
            # Transcribe audio
            await sio.emit('transcription_status', {
                'status': 'transcribing',
                'message': 'Transcribing audio...',
                'video_id': video_id,
                'progress': 50
            }, room=sid)
            
            transcription_result = await transcription_service.transcribe(
                audio_path, 
                video_id=video_id
            )
            
            if not transcription_result or not transcription_result.get('text'):
                raise Exception("Transcription failed or returned empty result")
            
            # Update status to completing
            await sio.emit('transcription_status', {
                'status': 'completing',
                'message': 'Processing transcription...',
                'video_id': video_id,
                'progress': 90
            }, room=sid)
            
            # Cache the transcription
            if video_id:
                await transcription_service.cache_transcription(
                    video_id, 
                    transcription_result
                )
            
            # Send the transcription data
            await sio.emit('transcription_data', {
                'status': 'success',
                'video_id': video_id,
                'transcript': transcription_result['text'],
                'has_timestamps': transcription_result.get('has_timestamps', False)
            }, room=sid)
            
            # Set transcription length for UI
            await sio.emit('transcription_length', {
                'video_id': video_id,
                'length': len(transcription_result['text'])
            }, room=sid)
            
            # Update completed status
            await sio.emit('transcription_status', {
                'status': 'completed',
                'message': 'Transcription completed',
                'video_id': video_id,
                'progress': 100
            }, room=sid)
            
        except asyncio.CancelledError:
            logger.info(f"Transcription process for {video_id or video_url} was cancelled")
            await sio.emit('transcription_error', {
                'error': 'Operation cancelled',
                'status': 'cancelled',
                'video_id': video_id
            }, room=sid)
        except Exception as e:
            logger.error(f"Error in transcription process: {str(e)}\n{traceback.format_exc()}")
            await sio.emit('transcription_error', {
                'error': str(e),
                'status': 'error',
                'video_id': video_id
            }, room=sid)
    
    try:
        # Create task and register it
        if sid not in active_tasks:
            active_tasks[sid] = {}
        
        task = asyncio.create_task(process_transcription())
        active_tasks[sid]['transcription'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout in transcription for {video_id or video_url}")
            await sio.emit('transcription_error', {
                'error': 'Operation timed out',
                'status': 'timeout',
                'video_id': video_id
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'transcription' in active_tasks[sid]:
                del active_tasks[sid]['transcription']
    
    except Exception as e:
        logger.error(f"Error setting up transcription task: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('transcription_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

@sio.event
async def get_timestamps(sid, data):
    """
    Handle requests to get video timestamps
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    
    logger.info(f"Received get_timestamps request from {client_id} for video: {video_id}")
    
    if not video_id:
        await sio.emit('timestamps_error', {
            'error': 'Missing video_id',
            'status': 'error'
        }, room=sid)
        return
    
    try:
        # Create task for getting timestamps with timeout
        async def get_timestamps_task():
            try:
                # Get timestamps from the service
                timestamps = await transcription_service.get_timestamps(video_id)
                
                if not timestamps:
                    timestamps = []
                
                # Send timestamps data
                await sio.emit('timestamps_data', {
                    'timestamps': timestamps,
                    'video_id': video_id
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error getting timestamps: {str(e)}")
                await sio.emit('timestamps_error', {
                    'error': str(e),
                    'status': 'error',
                    'video_id': video_id
                }, room=sid)
        
        # Create the task with timeout
        task = asyncio.create_task(get_timestamps_task())
        
        # Register task
        if sid not in active_tasks:
            active_tasks[sid] = {}
        active_tasks[sid]['get_timestamps'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting timestamps for {video_id}")
            await sio.emit('timestamps_error', {
                'error': 'Operation timed out',
                'status': 'timeout',
                'video_id': video_id
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'get_timestamps' in active_tasks[sid]:
                del active_tasks[sid]['get_timestamps']
    
    except Exception as e:
        logger.error(f"Error in get_timestamps: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('timestamps_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

@sio.event
async def get_topics(sid, data):
    """
    Handle requests to get video topics with timeout protection
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    
    logger.info(f"Received get_topics request from {client_id} for video: {video_id}")
    
    if not video_id:
        await sio.emit('topics_error', {
            'error': 'Missing video_id',
            'status': 'error'
        }, room=sid)
        return
    
    try:
        # Create an async task to process topics with timeout
        async def process_topics():
            try:
                # Get topics from the service
                topics = await transcription_service.get_topics(video_id)
                
                if not topics:
                    topics = []
                
                # Send topics data
                await sio.emit('topics_data', {
                    'topics': topics,
                    'video_id': video_id,
                    'status': 'success'
                }, room=sid)
                
            except asyncio.CancelledError:
                logger.info(f"Topics processing for {video_id} was cancelled")
            except Exception as e:
                logger.error(f"Error processing topics: {str(e)}")
                await sio.emit('topics_error', {
                    'error': str(e),
                    'status': 'error',
                    'video_id': video_id
                }, room=sid)
        
        # Create the task with timeout
        task = asyncio.create_task(process_topics())
        
        # Register task
        if sid not in active_tasks:
            active_tasks[sid] = {}
        active_tasks[sid]['get_topics'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting video topics for {video_id}")
            # Send empty array with timeout status
            await sio.emit('topics_data', {
                'topics': [],
                'video_id': video_id,
                'status': 'timeout'
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'get_topics' in active_tasks[sid]:
                del active_tasks[sid]['get_topics']
        
    except Exception as e:
        logger.error(f"Error in get_topics: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('topics_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

@sio.event
async def get_scenes(sid, data):
    """
    Handle requests to get video scenes
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    
    logger.info(f"Received get_scenes request from {client_id} for video: {video_id}")
    
    if not video_id:
        await sio.emit('scenes_error', {
            'error': 'Missing video_id',
            'status': 'error'
        }, room=sid)
        return
    
    try:
        # Create task for getting scenes with timeout
        async def get_scenes_task():
            try:
                # Get scenes from the service
                scenes = await video_processor.get_scenes(video_id)
                
                if not scenes:
                    scenes = []
                
                # Send scenes data
                await sio.emit('scenes_data', {
                    'scenes': scenes,
                    'video_id': video_id,
                    'status': 'success'
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error getting scenes: {str(e)}")
                await sio.emit('scenes_error', {
                    'error': str(e),
                    'status': 'error',
                    'video_id': video_id
                }, room=sid)
        
        # Create the task with timeout
        task = asyncio.create_task(get_scenes_task())
        
        # Register task
        if sid not in active_tasks:
            active_tasks[sid] = {}
        active_tasks[sid]['get_scenes'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting scenes for {video_id}")
            await sio.emit('scenes_error', {
                'error': 'Operation timed out',
                'status': 'timeout',
                'video_id': video_id
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'get_scenes' in active_tasks[sid]:
                del active_tasks[sid]['get_scenes']
        
    except Exception as e:
        logger.error(f"Error in get_scenes: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('scenes_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

@sio.event
async def get_highlights(sid, data):
    """
    Handle requests to get video highlights with timeout protection
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    
    logger.info(f"Received get_highlights request from {client_id} for video: {video_id}")
    
    if not video_id:
        await sio.emit('highlights_error', {
            'error': 'Missing video_id',
            'status': 'error'
        }, room=sid)
        return
    
    try:
        # Create an async task to process highlights with timeout
        async def process_highlights():
            try:
                # Get highlights from the service
                highlights = await transcription_service.get_highlights(video_id)
                
                if not highlights:
                    highlights = []
                
                # Send highlights data
                await sio.emit('highlights_data', {
                    'highlights': highlights,
                    'video_id': video_id,
                    'status': 'success'
                }, room=sid)
                
            except asyncio.CancelledError:
                logger.info(f"Highlights processing for {video_id} was cancelled")
            except Exception as e:
                logger.error(f"Error processing highlights: {str(e)}")
                await sio.emit('highlights_error', {
                    'error': str(e),
                    'status': 'error',
                    'video_id': video_id
                }, room=sid)
        
        # Create the task with timeout
        task = asyncio.create_task(process_highlights())
        
        # Register task
        if sid not in active_tasks:
            active_tasks[sid] = {}
        active_tasks[sid]['get_highlights'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting video highlights for {video_id}")
            # Send empty array with timeout status
            await sio.emit('highlights_data', {
                'highlights': [],
                'video_id': video_id,
                'status': 'timeout'
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'get_highlights' in active_tasks[sid]:
                del active_tasks[sid]['get_highlights']
        
    except Exception as e:
        logger.error(f"Error in get_highlights: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('highlights_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

@sio.event
async def get_frames(sid, data):
    """
    Handle requests to get video frames
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    
    logger.info(f"Received get_frames request from {client_id} for video: {video_id}")
    
    if not video_id:
        await sio.emit('frames_error', {
            'error': 'Missing video_id',
            'status': 'error'
        }, room=sid)
        return
    
    try:
        # Create task for getting frames with timeout
        async def get_frames_task():
            try:
                # Get frames from the service
                frames = await video_processor.get_frames(video_id)
                
                if not frames:
                    frames = []
                
                # Send frames data
                await sio.emit('frames_data', {
                    'frames': frames,
                    'video_id': video_id,
                    'status': 'success'
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error getting frames: {str(e)}")
                await sio.emit('frames_error', {
                    'error': str(e),
                    'status': 'error',
                    'video_id': video_id
                }, room=sid)
        
        # Create the task with timeout
        task = asyncio.create_task(get_frames_task())
        
        # Register task
        if sid not in active_tasks:
            active_tasks[sid] = {}
        active_tasks[sid]['get_frames'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting frames for {video_id}")
            await sio.emit('frames_error', {
                'error': 'Operation timed out',
                'status': 'timeout',
                'video_id': video_id
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'get_frames' in active_tasks[sid]:
                del active_tasks[sid]['get_frames']
        
    except Exception as e:
        logger.error(f"Error in get_frames: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('frames_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

@sio.event
async def analyze_visual(sid, data):
    """
    Handle requests to analyze visual content
    """
    client_id = session_data.get(sid, {}).get('client_id', 'unknown')
    video_id = data.get('video_id')
    
    logger.info(f"Received analyze_visual request from {client_id} for video: {video_id}")
    
    if not video_id:
        await sio.emit('visual_error', {
            'error': 'Missing video_id',
            'status': 'error'
        }, room=sid)
        return
    
    try:
        # Create task for visual analysis with timeout
        async def analyze_visual_task():
            try:
                # Get visual analysis from the service
                visual_data = await video_processor.analyze_visual(video_id)
                
                if not visual_data:
                    visual_data = {
                        "frames": [],
                        "scenes": [],
                        "objects": [],
                        "highlights": []
                    }
                
                # Send visual data
                await sio.emit('visual_data', {
                    'visual': visual_data,
                    'video_id': video_id,
                    'status': 'success'
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error in visual analysis: {str(e)}")
                await sio.emit('visual_error', {
                    'error': str(e),
                    'status': 'error',
                    'video_id': video_id
                }, room=sid)
        
        # Create the task with timeout
        task = asyncio.create_task(analyze_visual_task())
        
        # Register task
        if sid not in active_tasks:
            active_tasks[sid] = {}
        active_tasks[sid]['analyze_visual'] = task
        
        # Set up timeout
        try:
            await asyncio.wait_for(task, timeout=OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout in visual analysis for {video_id}")
            await sio.emit('visual_error', {
                'error': 'Operation timed out',
                'status': 'timeout',
                'video_id': video_id
            }, room=sid)
        finally:
            # Clean up the task
            if sid in active_tasks and 'analyze_visual' in active_tasks[sid]:
                del active_tasks[sid]['analyze_visual']
        
    except Exception as e:
        logger.error(f"Error in analyze_visual: {str(e)}\n{traceback.format_exc()}")
        await sio.emit('visual_error', {
            'error': str(e),
            'status': 'error',
            'video_id': video_id
        }, room=sid)

if __name__ == "__main__":
    import uvicorn
    
    print("Starting standalone Socket.IO server on http://0.0.0.0:8001")
    print("Test URL: http://localhost:8001/socket.io/?EIO=4&transport=polling")
    
    uvicorn.run(socket_app, host="0.0.0.0", port=8001)