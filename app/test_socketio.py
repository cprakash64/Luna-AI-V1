# test_socketio.py
import socketio
import uvicorn
from fastapi import FastAPI
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("socketio")

# Create a FastAPI app
app = FastAPI()

# Create a Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=["*"],  # Allow all origins in development
    logger=True,
    engineio_logger=True
)

# Create an ASGI app for Socket.IO
socket_app = socketio.ASGIApp(sio)

# Mount Socket.IO at /socket.io
app.mount("/socket.io", socket_app)

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    await sio.emit('welcome', {'message': 'Connected to Socket.IO server'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def echo(sid, data):
    logger.info(f"Received echo from {sid}: {data}")
    await sio.emit('echo_response', {'data': data, 'sid': sid}, room=sid)

if __name__ == "__main__":
    print("Starting Socket.IO server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)