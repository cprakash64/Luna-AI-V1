"""
app/api/timestamps.py
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import logging
import json
import os

from app.utils.database import get_db
from app.config import settings

# Create a logger
logger = logging.getLogger(__name__)

# Create a router for timestamp-related endpoints
router = APIRouter()

# Define get_optional_auth directly in this file to avoid import issues
async def get_optional_auth(request: Request = None, db: Session = Depends(get_db)):
    """
    In development mode, bypass authentication for easier testing
    In production, require proper authentication
    
    This is a standalone version that doesn't require importing from auth.py
    """
    # Check if we should bypass auth (for development only)
    DEV_MODE = os.environ.get("DEV_MODE", "False").lower() in ("true", "1", "t")
    
    if DEV_MODE:
        logger.info("Using development mode authentication bypass")
        # Return a default testing user
        return {
            "id": "dev-user",
            "email": "dev@example.com",
            "username": "dev_user",
            "is_active": True,
            "authenticated": True
        }
    else:
        # In production, use normal auth
        try:
            from app.services.auth import get_current_user
            return await get_current_user(db=db)
        except Exception as e:
            logger.warning(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

async def read_timestamps_file(video_id: str) -> dict:
    """Read timestamps from JSON file"""
    try:
        # Check if timestamps file exists
        timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
        if os.path.exists(timestamp_path):
            with open(timestamp_path, "r") as f:
                return json.load(f)
        return {"formatted_timestamps": [], "raw_timestamps": []}
    except Exception as e:
        logger.error(f"Error reading timestamps file: {str(e)}")
        return {"formatted_timestamps": [], "raw_timestamps": []}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Luna AI"}

@router.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {"status": "ok", "service": "Luna AI"}

@router.get("/v1/health")
@router.get("/api/v1/health")
async def api_v1_health_check():
    """API v1 health check endpoint"""
    return {"status": "ok", "service": "Luna AI"}

@router.get("/api/videos/{video_id}/timestamps")
@router.get("/api/v1/videos/{video_id}/timestamps")
async def get_video_timestamps(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_optional_auth),
    db: Session = Depends(get_db)
):
    """Get timestamps for a specific video"""
    logger.info(f"Timestamps requested for video: {video_id}")
    
    try:
        # Read timestamps from file
        timestamps_data = await read_timestamps_file(video_id)
        formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
        
        # Convert to API response format
        response_timestamps = []
        for ts in formatted_timestamps:
            response_timestamps.append({
                "time": ts.get("start_time", 0),
                "timestamp": ts.get("start_time", 0),
                "text": ts.get("text", ""),
                "display_time": ts.get("display_time", "00:00"),
                "end_time": ts.get("end_time", 0)
            })
        
        return {
            "status": "success",
            "video_id": video_id,
            "timestamps": response_timestamps
        }
    except Exception as e:
        logger.error(f"Error getting timestamps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving timestamps: {str(e)}")

@router.get("/api/timestamps")
async def get_timestamps_by_query(
    videoId: str = Query(None),
    request: Request = None,
    current_user: Dict[str, Any] = Depends(get_optional_auth),
    db: Session = Depends(get_db)
):
    """Get timestamps by video ID as a query parameter"""
    if not videoId:
        raise HTTPException(status_code=400, detail="Missing videoId parameter")
    
    # Forward to the endpoint with video_id in path
    return await get_video_timestamps(videoId, request, current_user, db)