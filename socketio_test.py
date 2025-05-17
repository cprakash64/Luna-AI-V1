"""
Socket.IO Diagnostic Tool for Luna AI
Run this file to test Socket.IO connectivity
"""
import socketio
import asyncio
import logging
import sys
import json
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Function to create a Socket.IO client
def create_socket_client(server_url='http://localhost:8000'):
    """Create a Socket.IO client connected to the specified server"""
    # Create a standard Socket.IO client
    sio = socketio.AsyncClient(
        logger=True,
        engineio_logger=True
    )
    
    # Set up event handlers
    @sio.event
    async def connect():
        logger.info(f"Connected to server at {server_url}")
        return True
    
    @sio.event
    async def connect_error(data):
        logger.error(f"Connection error: {data}")
        return False
        
    @sio.event
    async def disconnect():
        logger.info("Disconnected from server")
    
    @sio.event
    async def connection_established(data):
        logger.info(f"Connection established event received: {data}")
    
    @sio.event
    async def echo_response(data):
        logger.info(f"Echo response received: {data}")
    
    @sio.event
    async def transcription_status(data):
        logger.info(f"Transcription status update: {data}")
    
    @sio.event
    async def transcription(data):
        logger.info(f"Transcription received: {data}")
    
    @sio.event
    async def error(data):
        logger.error(f"Error event from server: {data}")
    
    return sio

async def run_diagnostics():
    """Run a series of diagnostic tests for the Socket.IO connection"""
    # Generate a random client ID and tab ID for testing
    client_id = f"test_client_{uuid.uuid4().hex[:8]}"
    tab_id = f"test_tab_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Starting Socket.IO diagnostic with client ID: {client_id}, tab ID: {tab_id}")
    
    # Create Socket.IO client
    sio = create_socket_client()
    
    try:
        # Connect to server
        logger.info("Attempting to connect to server...")
        await sio.connect(
            'http://localhost:8000',
            headers={'X-Client-ID': client_id},
            transports=['websocket']
        )
        
        # Wait for connection to be established
        logger.info("Waiting for connection to stabilize...")
        await asyncio.sleep(1)
        
        # Test 1: Register tab
        logger.info("Test 1: Registering tab...")
        await sio.emit('register_tab', {'tabId': tab_id})
        await asyncio.sleep(1)
        
        # Test 2: Echo test
        logger.info("Test 2: Testing echo functionality...")
        await sio.emit('echo', {'message': 'Hello from diagnostic tool!'})
        await asyncio.sleep(1)
        
        # Test 3: YouTube URL processing 
        logger.info("Test 3: Testing YouTube URL processing...")
        await sio.emit('transcribe_youtube_video', {
            'video_id': tab_id,
            'youtube_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Test video URL
        })
        
        # Allow some time for processing updates to be received
        logger.info("Waiting for transcription status updates...")
        await asyncio.sleep(5)
        
        # Disconnect from server
        logger.info("Diagnostics complete, disconnecting...")
        await sio.disconnect()
        
    except Exception as e:
        logger.error(f"Error during diagnostics: {e}")
        if sio.connected:
            await sio.disconnect()
    
if __name__ == "__main__":
    logger.info("Socket.IO diagnostic tool for Luna AI")
    logger.info("This will test the Socket.IO connection to your server")
    
    # Run the diagnostic tests
    asyncio.run(run_diagnostics())