"""
Debugging script for video upload transcription issues
Add this to your project and run it to diagnose the problem
"""
import os
import json
import logging
import asyncio
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("upload_debug")

# Check if FFmpeg is installed and accessible
def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info(f"FFmpeg is installed: {result.stdout.splitlines()[0]}")
            return True
        else:
            logger.error(f"FFmpeg check failed with return code {result.returncode}: {result.stderr}")
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

# Check file existence and permissions
def check_file(path):
    """Check if a file exists and has correct permissions"""
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return False
        
    if not os.path.isfile(path):
        logger.error(f"Path is not a file: {path}")
        return False
        
    if not os.access(path, os.R_OK):
        logger.error(f"File is not readable: {path}")
        return False
        
    size = os.path.getsize(path)
    logger.info(f"File exists: {path}, Size: {size} bytes")
    return True

# Check directory permissions
def check_dir(path):
    """Check if a directory exists and has correct permissions"""
    if not os.path.exists(path):
        logger.error(f"Directory not found: {path}")
        return False
        
    if not os.path.isdir(path):
        logger.error(f"Path is not a directory: {path}")
        return False
        
    if not os.access(path, os.R_OK | os.W_OK):
        logger.error(f"Directory is not readable/writable: {path}")
        return False
        
    logger.info(f"Directory exists with correct permissions: {path}")
    return True

# Test audio extraction
async def test_audio_extraction(video_path):
    """Test audio extraction from video"""
    import tempfile
    import subprocess
    
    logger.info(f"Testing audio extraction from: {video_path}")
    
    if not check_file(video_path):
        return False
    
    try:
        # Extract audio to a temporary file
        audio_path = tempfile.mktemp(suffix=".wav")
        
        # Create the FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-ar", "16000",  # 16kHz sample rate for best compatibility
            "-ac", "1",      # Mono audio channel
            audio_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for the process
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg failed with return code {process.returncode}: {stderr.decode()}")
            return False
            
        # Check if audio file was created
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error(f"Failed to extract audio: output file is empty or missing")
            return False
            
        logger.info(f"Successfully extracted audio to: {audio_path} (size: {os.path.getsize(audio_path)} bytes)")
        
        # Clean up
        os.remove(audio_path)
        return True
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}", exc_info=True)
        return False

# Test task queue
async def test_task_queue():
    """Test task queue functionality"""
    from app.services.transcription_service import task_queue
    
    logger.info("Testing task queue...")
    
    try:
        # Check if task queue is running
        if not task_queue.is_running:
            logger.error("Task queue is not running")
            return False
        
        # Try to add a diagnostic task
        task_id = await task_queue.add_task(
            task_type="diagnostic",
            params={"test": True},
            priority=10
        )
        
        logger.info(f"Added diagnostic task {task_id} to queue")
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Check task status
        task = task_queue.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found in queue")
            return False
            
        logger.info(f"Task status: {task.status}")
        
        # Wait for completion
        for _ in range(10):  # Wait up to 10 seconds
            task = task_queue.get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} disappeared from queue")
                return False
                
            if task.status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
        
        if task.status == "completed":
            logger.info(f"Task completed successfully")
            return True
        else:
            logger.error(f"Task did not complete successfully. Status: {task.status}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing task queue: {str(e)}", exc_info=True)
        return False

# Test WebSocket communication
async def test_websocket(tab_id):
    """Test WebSocket communication"""
    from app.api.websockets import sio
    
    logger.info(f"Testing WebSocket communication with tab_id: {tab_id}")
    
    try:
        # Try to emit a test message
        await sio.emit('test_connection', {
            'status': 'test',
            'message': 'Testing WebSocket communication',
            'video_id': tab_id
        }, room=tab_id)
        
        logger.info(f"Emitted test message to tab_id: {tab_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing WebSocket: {str(e)}", exc_info=True)
        return False

# Run all tests
async def run_diagnostics(video_path, tab_id):
    """Run all diagnostic tests"""
    results = {}
    
    # Check FFmpeg
    results["ffmpeg_installed"] = check_ffmpeg()
    
    # Check directories
    from app.config import settings
    results["upload_dir_ok"] = check_dir(settings.UPLOAD_DIR)
    results["transcription_dir_ok"] = check_dir(settings.TRANSCRIPTION_DIR)
    results["temp_dir_ok"] = check_dir(settings.TEMP_DIR)
    
    # Check video file
    results["video_file_ok"] = check_file(video_path)
    
    # Test audio extraction
    if results["video_file_ok"] and results["ffmpeg_installed"]:
        results["audio_extraction_ok"] = await test_audio_extraction(video_path)
    else:
        results["audio_extraction_ok"] = False
    
    # Test task queue
    results["task_queue_ok"] = await test_task_queue()
    
    # Test WebSocket
    results["websocket_ok"] = await test_websocket(tab_id)
    
    # Print results
    logger.info("=== DIAGNOSTIC RESULTS ===")
    for test, result in results.items():
        logger.info(f"{test}: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Suggest fixes
    logger.info("=== SUGGESTED FIXES ===")
    if not results["ffmpeg_installed"]:
        logger.info("* Install FFmpeg and make sure it's in your system PATH")
    
    if not results["upload_dir_ok"] or not results["transcription_dir_ok"] or not results["temp_dir_ok"]:
        logger.info("* Check directory permissions and make sure they exist")
    
    if not results["video_file_ok"]:
        logger.info("* Check that the uploaded video file exists and is accessible")
    
    if not results["audio_extraction_ok"]:
        logger.info("* Try a different video format or check FFmpeg compatibility")
    
    if not results["task_queue_ok"]:
        logger.info("* Check task queue configuration and make sure handlers are registered")
    
    if not results["websocket_ok"]:
        logger.info("* Check WebSocket server configuration and client connection")
    
    return results

# Execute this script with a specific video and tab_id
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose video upload transcription issues")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--tab_id", required=True, help="Tab ID for testing")
    
    args = parser.parse_args()
    
    asyncio.run(run_diagnostics(args.video, args.tab_id))