# app/socketio_adapter.py

"""
Socket.IO adapter for Luna AI
Provides WebSocket communication for chat
"""
import logging
import socketio
from fastapi import FastAPI
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class SocketManager:
    """
    Socket.IO manager for Luna AI
    Handles WebSocket connections and events
    """
    
    def __init__(self):
        """Initialize the socket manager"""
        self.sio = None
        self.socket_app = None
    
    def create_socket(self):
        """Create and configure the Socket.IO server"""
        logger.info("Creating Socket.IO server with CORS origin: *")
        
        # Create Socket.IO server with allow-all CORS - fix: use string, not list
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins='*',  # Fixed: String instead of list
            logger=True,
            engineio_logger=True,
            ping_timeout=120,  # Increased from 60
            ping_interval=40,   # Increased from 25
        )
        
        # Create ASGI app - fix: use consistent socketio_path
        self.socket_app = socketio.ASGIApp(
            self.sio,
            socketio_path='socket.io',  # Fix: Match the mount path
            static_files={},
            other_asgi_app=None,
        )
        
        # Register event handlers
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"Client connected: {sid}")
            await self.sio.emit('welcome', {'message': 'Connected to Luna AI'}, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {sid}")
        
        @self.sio.event
        async def ask_ai(sid, data):
            """Handle AI question from client"""
            try:
                logger.info(f"Received question from {sid}: {data}")
                
                question = data.get('question', '')
                tab_id = data.get('tabId', '')
                
                # Simple echo response for now
                response = f"Echo: {question}"
                
                # Send response back to client
                await self.sio.emit('ai_response', {'answer': response}, room=sid)
                
            except Exception as e:
                logger.error(f"Error in ask_ai: {e}")
                await self.sio.emit('ai_response', {'error': str(e)}, room=sid)
        
        @self.sio.event
        async def echo(sid, data):
            """Echo event for testing"""
            logger.info(f"Received echo from {sid}: {data}")
            await self.sio.emit('echo_response', {
                'status': 'success',
                'message': 'Echo received',
                'received_data': data,
                'sid': sid
            }, room=sid)
            
        # Add missing handlers for events in your error logs
        @self.sio.event
        async def get_video_timestamps(sid, data):
            """Handle video timestamps request"""
            try:
                video_id = data.get('videoId', '')
                logger.info(f"Fetching timestamps for video: {video_id}")
                # Your timestamp logic here
                await self.sio.emit('video_timestamps', {'timestamps': []}, room=sid)
            except Exception as e:
                logger.error(f"Error getting timestamps: {e}")
                await self.sio.emit('video_timestamps', {'error': str(e)}, room=sid)
                
        @self.sio.event
        async def get_video_scenes(sid, data):
            """Handle video scenes request"""
            try:
                video_id = data.get('videoId', '')
                logger.info(f"Fetching scenes for video: {video_id}")
                # Your scenes logic here
                await self.sio.emit('video_scenes', {'scenes': []}, room=sid)
            except Exception as e:
                logger.error(f"Error getting scenes: {e}")
                await self.sio.emit('video_scenes', {'error': str(e)}, room=sid)
                
        @self.sio.event
        async def get_video_topics(sid, data):
            """Handle video topics request"""
            try:
                video_id = data.get('videoId', '')
                logger.info(f"Fetching topics for video: {video_id}")
                # Your topics logic here
                await self.sio.emit('video_topics', {'topics': []}, room=sid)
            except Exception as e:
                logger.error(f"Error getting topics: {e}")
                await self.sio.emit('video_topics', {'error': str(e)}, room=sid)
                
        @self.sio.event
        async def get_video_frames(sid, data):
            """Handle video frames request"""
            try:
                video_id = data.get('videoId', '')
                logger.info(f"Fetching frames for video: {video_id}")
                # Your frames logic here
                await self.sio.emit('video_frames', {'frames': []}, room=sid)
            except Exception as e:
                logger.error(f"Error getting frames: {e}")
                await self.sio.emit('video_frames', {'error': str(e)}, room=sid)
        
        return self.socket_app
    
    def mount_to_app(self, app: FastAPI, path: str = "/socket.io"):
        """Mount the Socket.IO app to a FastAPI app"""
        if not self.socket_app:
            self.create_socket()
        
        # Mount the Socket.IO ASGI app to the FastAPI app
        app.mount(path, self.socket_app)
        logger.info(f"Socket.IO mounted at {path}")
        
        # NOTE: Don't add middleware to Socket.IO app - it doesn't support FastAPI middleware
        # Instead, CORS is handled by cors_allowed_origins in the socketio.AsyncServer initialization

# Create a global instance
socket_manager = SocketManager()