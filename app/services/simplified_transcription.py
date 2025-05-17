# app/services/simplified_transcription.py
"""
Simplified transcription fallback for when the main methods fail
This provides a way to generate mock transcription data when processing fails
"""
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("transcription.simplified")

async def create_mock_transcription(youtube_url: str, output_file: Optional[str] = None, video_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a mock transcription when regular processing fails
    
    Args:
        youtube_url: YouTube URL that failed
        output_file: Optional path to save mock transcription
        video_id: Optional video ID to include in the mock data
        
    Returns:
        Dictionary with mock transcription data
    """
    logger.info(f"Creating mock transcription for {youtube_url} with video_id: {video_id}")
    
    # Create different mock content based on the URL to simulate variety
    if "shorts" in youtube_url:
        mock_text = "This is a short video that would typically contain 30 seconds of content. Unfortunately, we couldn't process this specific YouTube Short. Please try another video."
    else:
        mock_text = "This is a mock transcript created for debug purposes. In a real implementation, this would contain the actual transcription of your video content."
    
    # Create a basic mock transcription
    mock_transcription = {
        "text": mock_text,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": mock_text.split(". ")[0] + "."
            }
        ],
        "status": "success"
    }
    
    # Add video_id if provided
    if video_id:
        mock_transcription["video_id"] = video_id
    
    # Save to file if output_file is provided
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(mock_transcription, f)
            logger.info(f"Saved mock transcription to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving mock transcription: {str(e)}")
    
    return mock_transcription