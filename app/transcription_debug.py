"""
Debug helper for transcription service
"""
import os
import logging

# Configure logging
logger = logging.getLogger("transcription.debug")

# Enable debug transcription mode
def enable_debug_transcription():
    """
    Enable debug transcription mode
    This uses mock transcription data instead of calling external services
    """
    os.environ["DEBUG_TRANSCRIPTION"] = "1"
    logger.info("Debug transcription mode ENABLED")
    return True

# Disable debug transcription mode
def disable_debug_transcription():
    """
    Disable debug transcription mode
    This will use the real transcription services
    """
    os.environ["DEBUG_TRANSCRIPTION"] = "0"
    logger.info("Debug transcription mode DISABLED")
    return False

# Check if debug transcription mode is enabled
def is_debug_transcription_enabled():
    """
    Check if debug transcription mode is enabled
    """
    return os.environ.get("DEBUG_TRANSCRIPTION") == "1"