"""
API endpoints for object detection and frame analysis
"""
import base64
import logging
import io
import numpy as np
import cv2
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Body, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional

from app.services.object_detection import get_detector, analyze_frames_for_video
from app.services.clip_service import get_clip_service
from app.utils.database import get_db_context
from app.dependencies import get_current_user

# Configure logging
logger = logging.getLogger("api.object_detection")

# Create router
router = APIRouter()

@router.post("/detect-objects")
async def detect_objects_in_image(
    data: Dict[str, Any] = Body(...),
    current_user = Depends(get_current_user)
):
    """
    Detect objects in an image
    
    The image can be provided as a base64-encoded string
    
    Returns:
        List of detected objects with bounding boxes and colors
    """
    try:
        # Check if image data is provided
        if "image" not in data or not data["image"]:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Parse image data
        image_data = data["image"]
        
        # Handle base64 encoded image
        if image_data.startswith("data:image"):
            # Extract base64 content
            image_data = image_data.split(",", 1)[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get detector
        detector = get_detector()
        
        # Detect objects
        detections = detector.detect_objects(image)
        
        # Add color analysis
        for detection in detections:
            # Ensure we have box coordinates
            if "box" in detection and len(detection["box"]) == 4:
                x1, y1, x2, y2 = detection["box"]
                # Extract object region
                object_roi = image[y1:y2, x1:x2]
                
                # Skip if ROI is empty
                if object_roi.size == 0 or object_roi.shape[0] == 0 or object_roi.shape[1] == 0:
                    continue
                
                # Get dominant color
                colors = detector.color_recognizer.extract_colors(image, detection["box"])
                if colors:
                    detection["colors"] = colors
                    detection["dominant_color"] = colors[0][0]
        
        # Add CLIP description if available
        clip_service = get_clip_service()
        if clip_service.is_available():
            description = clip_service.describe_image(image)
            
            # Return results with CLIP description
            return {
                "objects": detections,
                "scene_type": description.get("scene_type", "unknown"),
                "attributes": description.get("attributes", []),
                "timestamp": datetime.now().isoformat()
            }
        
        # Return results without CLIP description
        return {
            "objects": detections,
            "timestamp": datetime.now().isoformat()
        }
            
    except Exception as e:
        logger.error(f"Error detecting objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")

@router.post("/upload-frame")
async def upload_frame_for_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    timestamp: float = Form(0.0),
    video_id: str = Form(...),
    current_user = Depends(get_current_user)
):
    """
    Upload a video frame for analysis
    
    The frame is analyzed for objects, colors, and scene content
    
    Returns:
        Analysis results for the frame
    """
    try:
        # Read frame data
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get detector and CLIP service
        detector = get_detector()
        clip_service = get_clip_service()
        
        # Analyze frame
        analysis = detector.analyze_frame(frame)
        
        # Add CLIP description if available
        if clip_service.is_available():
            clip_description = clip_service.describe_image(frame)
            analysis["scene_type"] = clip_description.get("scene_type", "unknown")
            analysis["attributes"] = clip_description.get("attributes", [])
        
        # Save analysis to database in background
        background_tasks.add_task(
            save_frame_analysis,
            video_id,
            timestamp,
            analysis
        )
        
        # Return analysis results
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": timestamp,
            "video_id": video_id
        }
            
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

@router.get("/video/{video_id}/frames")
async def get_video_frames(
    video_id: str,
    count: int = 10,
    current_user = Depends(get_current_user)
):
    """
    Get analyzed frames for a video
    
    Returns:
        List of analyzed frames with timestamps
    """
    try:
        with get_db_context() as db:
            # Query for frames
            query = """
            SELECT id, timestamp, objects, scene_colors
            FROM frames
            WHERE video_id = %s
            ORDER BY timestamp
            LIMIT %s
            """
            
            result = db.execute(query, (video_id, count))
            frames = []
            
            for row in result:
                frames.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "timestamp_formatted": format_timestamp(row[1]),
                    "objects": row[2],
                    "colors": row[3]
                })
            
            return {"frames": frames, "video_id": video_id}
            
    except Exception as e:
        logger.error(f"Error fetching video frames: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching video frames: {str(e)}")

@router.post("/video/{video_id}/analyze-frame/{timestamp}")
async def analyze_video_frame_at_timestamp(
    video_id: str,
    timestamp: float,
    data: Dict[str, Any] = Body(...),
    current_user = Depends(get_current_user)
):
    """
    Analyze a specific frame at a timestamp
    
    Returns:
        Analysis results for the frame
    """
    try:
        # Check if frame data is provided
        if "frame" not in data or not data["frame"]:
            raise HTTPException(status_code=400, detail="No frame data provided")
        
        # Parse frame data
        frame_data = data["frame"]
        
        # Handle base64 encoded image
        if frame_data.startswith("data:image"):
            # Extract base64 content
            frame_data = frame_data.split(",", 1)[1]
        
        # Decode base64 to image
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Get detector and CLIP service
        detector = get_detector()
        clip_service = get_clip_service()
        
        # Analyze frame
        analysis = detector.analyze_frame(frame)
        
        # Add CLIP description if available
        clip_description = None
        if clip_service.is_available():
            clip_description = clip_service.describe_image(frame)
            analysis["scene_type"] = clip_description.get("scene_type", "unknown")
            analysis["attributes"] = clip_description.get("attributes", [])
        
        # Save analysis to database
        save_frame_analysis(video_id, timestamp, analysis)
        
        # Return analysis results
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": timestamp,
            "timestamp_formatted": format_timestamp(timestamp),
            "video_id": video_id,
            "clip_description": clip_description
        }
            
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

async def save_frame_analysis(
    video_id: str,
    timestamp: float,
    analysis: Dict[str, Any]
):
    """
    Save frame analysis to database
    
    Args:
        video_id: ID of the video
        timestamp: Timestamp in seconds
        analysis: Analysis results
    """
    try:
        with get_db_context() as db:
            # Convert data to JSON strings
            import json
            objects_json = json.dumps(analysis.get("detections", []))
            colors_json = json.dumps(analysis.get("scene_colors", []))
            object_counts_json = json.dumps(analysis.get("object_counts", {}))
            
            # Check if entry already exists
            check_query = """
            SELECT id FROM frames
            WHERE video_id = %s AND timestamp = %s
            """
            
            result = db.execute(check_query, (video_id, timestamp))
            existing_id = result.fetchone()
            
            if existing_id:
                # Update existing entry
                update_query = """
                UPDATE frames
                SET objects = %s, scene_colors = %s, object_counts = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                
                db.execute(update_query, (
                    objects_json,
                    colors_json,
                    object_counts_json,
                    existing_id[0]
                ))
            else:
                # Insert new entry
                insert_query = """
                INSERT INTO frames (video_id, timestamp, objects, scene_colors, object_counts)
                VALUES (%s, %s, %s, %s, %s)
                """
                
                db.execute(insert_query, (
                    video_id,
                    timestamp,
                    objects_json,
                    colors_json,
                    object_counts_json
                ))
            
            db.commit()
            
    except Exception as e:
        logger.error(f"Error saving frame analysis: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"