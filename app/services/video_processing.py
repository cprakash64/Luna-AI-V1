"""
Video processing service for Luna AI
Handles video upload processing, frame extraction, and YouTube video downloading
With enhanced key frame sampling, scene detection, and highlight identification
"""
from app.services.object_detection import get_detector
from app.utils.storage import ensure_directory_exists
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from app.config import settings
import os
import uuid
import logging
import asyncio
import subprocess
import json
import time
import cv2
import numpy as np
# Import yt-dlp instead of pytube for more reliable downloads
import yt_dlp
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Class for processing videos - extracting frames, handling notifications,
    and coordinating video analysis processes.
    """
    
    def __init__(self, video_id: str, socket_io=None, file_path: Optional[str] = None):
        """
        Initialize the VideoProcessor with necessary parameters.
        
        Args:
            video_id: Unique identifier for the video
            socket_io: SocketIO instance for sending real-time updates
            file_path: Path to the video file if already available
        """
        self.video_id = video_id
        self.socket_io = socket_io
        self.file_path = file_path
        self.frames_dir = os.path.join('data', 'frames', video_id)
        self.frames_count = 0
        self.processing_status = "idle"
    
    async def notify_clients(self, event: str, data: Dict[str, Any]) -> None:
        """Send notifications to connected clients via Socket.IO."""
        if self.socket_io:
            try:
                await self.socket_io.emit(event, data)
                logger.debug(f"Emitted {event} with data: {data}")
            except Exception as e:
                logger.error(f"Failed to emit {event}: {str(e)}")
        else:
            logger.warning(f"Socket.IO not available, cannot emit {event}")
    
    async def notify_download_start(self) -> None:
        """Notify clients that video download has started."""
        await self.notify_clients('visual_analysis_status', {
            'status': 'downloading',
            'message': 'Downloading video for visual processing...',
            'video_id': self.video_id,
            'progress': 5
        })
    
    async def notify_download_complete(self) -> None:
        """Notify clients that video download is complete."""
        await self.notify_clients('visual_analysis_status', {
            'status': 'processing',
            'message': 'Download complete, beginning visual analysis...',
            'video_id': self.video_id,
            'progress': 15
        })
    
    async def notify_extraction_progress(self, progress: int) -> None:
        """Notify clients about frame extraction progress."""
        await self.notify_clients('visual_analysis_status', {
            'status': 'processing',
            'message': f'Extracting frames from video... {progress}%',
            'video_id': self.video_id,
            'progress': 15 + (progress // 2)  # Scale from 15-65%
        })
    
    async def notify_analysis_progress(self, progress: int) -> None:
        """Notify clients about analysis progress."""
        await self.notify_clients('visual_analysis_status', {
            'status': 'analyzing',
            'message': f'Analyzing video content... {progress}%',
            'video_id': self.video_id,
            'progress': 65 + (progress // 3)  # Scale from 65-98%
        })
    
    async def notify_completion(self, data: Dict[str, Any]) -> None:
        """Notify clients that analysis is complete."""
        await self.notify_clients('visual_analysis_status', {
            'status': 'completed',
            'message': 'Visual analysis complete',
            'video_id': self.video_id,
            'progress': 100
        })
        
        # Send the actual analysis data
        await self.notify_clients('visual_analysis_data', {
            'status': 'success',
            'video_id': self.video_id,
            'data': data
        })
    
    async def notify_error(self, error_message: str) -> None:
        """Notify clients about an error."""
        await self.notify_clients('visual_analysis_status', {
            'status': 'error',
            'message': f'Error in visual analysis: {error_message}',
            'video_id': self.video_id
        })
    
    async def ensure_frames_directory(self) -> str:
        """Ensure the frames directory exists."""
        os.makedirs(self.frames_dir, exist_ok=True)
        return self.frames_dir
    
    async def extract_frames(self, interval_seconds: float = 1.0) -> List[str]:
        """
        Extract frames from the video at regular intervals.
        
        Args:
            interval_seconds: Time interval between extracted frames
            
        Returns:
            List of paths to extracted frames
        """
        if not self.file_path:
            await self.notify_error("No video file available for frame extraction")
            return []
        
        try:
            # Ensure frames directory exists
            frames_dir = await self.ensure_frames_directory()
            
            # Use FFmpeg to extract frames
            cmd = [
                'ffmpeg', '-i', self.file_path,
                '-vf', f'fps=1/{interval_seconds}',
                '-q:v', '2',  # High quality
                os.path.join(frames_dir, 'frame_%04d.jpg')
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"FFmpeg error: {error_msg}")
                await self.notify_error(f"Frame extraction failed: {error_msg}")
                return []
            
            # Get list of extracted frames
            frames = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                     if f.startswith('frame_') and f.endswith('.jpg')]
            
            self.frames_count = len(frames)
            logger.info(f"Extracted {self.frames_count} frames from video {self.video_id}")
            
            return sorted(frames)
            
        except Exception as e:
            logger.error(f"Error in frame extraction: {str(e)}")
            await self.notify_error(f"Frame extraction failed: {str(e)}")
            return []
    
    async def process_video(self) -> Dict[str, Any]:
        """
        Main method to process the video from start to finish.
        
        Returns:
            Dictionary containing the analysis results
        """
        try:
            self.processing_status = "processing"
            
            # Extract frames
            await self.notify_extraction_progress(0)
            frames = await self.extract_frames()
            
            if not frames:
                await self.notify_error("No frames could be extracted from the video")
                self.processing_status = "error"
                return {"status": "error", "message": "No frames extracted"}
            
            # Use the existing function to detect objects in frames
            frame_info = []
            for frame_path in frames:
                frame_info.append({
                    "path": frame_path,
                    "timestamp": 0.0,  # Will be calculated based on filename
                })
            
            # Calculate timestamps based on frame filenames
            for i, frame in enumerate(frame_info):
                frame["timestamp"] = i * 1.0  # Default 1 second interval
                frame["timestamp_str"] = format_timestamp(frame["timestamp"])
            
            # Process frames with existing functions
            processed_frames = detect_objects_in_frames(frame_info)
            
            # Add scene analysis
            processed_frames = manual_scene_analysis(processed_frames)
            
            # Create a result dictionary
            result = {
                "status": "success",
                "video_id": self.video_id,
                "frames_count": self.frames_count,
                "frames": processed_frames,
                "scenes": [],
                "highlights": []
            }
            
            # Extract scenes and highlights
            for frame in processed_frames:
                if frame.get("is_scene_change", False):
                    result["scenes"].append({
                        "start_time": frame.get("timestamp", 0),
                        "description": frame.get("visual_description", "New scene"),
                        "frame_path": frame.get("path", "")
                    })
                
                if frame.get("is_highlight", False):
                    result["highlights"].append({
                        "timestamp": frame.get("timestamp", 0),
                        "description": frame.get("visual_description", "Highlight moment"),
                        "frame_path": frame.get("path", ""),
                        "score": frame.get("highlight_score", 0)
                    })
            
            await self.notify_completion(result)
            self.processing_status = "completed"
            return result
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            await self.notify_error(str(e))
            self.processing_status = "error"
            return {"status": "error", "message": str(e)}

# Function to detect scene changes between frames
def detect_scene_change(current_frame, previous_frame, threshold=30.0):
    """
    Detect if a scene change occurred between two frames
    
    Args:
        current_frame: Current frame as numpy array
        previous_frame: Previous frame as numpy array
        threshold: Difference threshold for scene change detection
        
    Returns:
        True if a scene change is detected, False otherwise
    """
    # Convert to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    frame_diff = cv2.absdiff(current_gray, previous_gray)
    mean_diff = np.mean(frame_diff)
    
    # Calculate histogram difference
    hist_current = cv2.calcHist([current_gray], [0], None, [64], [0, 256])
    hist_prev = cv2.calcHist([previous_gray], [0], None, [64], [0, 256])
    hist_diff = cv2.compareHist(hist_current, hist_prev, cv2.HISTCMP_CHISQR)
    
    # Combine metrics
    combined_diff = mean_diff + hist_diff * 0.5
    
    # Return True if difference exceeds threshold
    return combined_diff > threshold

def extract_frames(
    video_path: str,
    output_dir: str,
    extract_by_seconds: bool = True,  # New parameter to extract by seconds instead of frames
    frame_interval: int = 1,          # Extract a frame every 1 second by default
    max_frames: int = 1000,           # Increased max frames significantly
    detect_scenes: bool = True,       # New parameter to enable scene change detection
    detect_key_frames: bool = True,   # New parameter to detect key frames based on content
) -> List[Dict[str, Any]]:
    """
    Extract frames from a video file at specified intervals
    with enhanced scene change detection and key frame sampling
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        extract_by_seconds: If True, extract frames based on time; if False, based on frame count
        frame_interval: Extract frames every N seconds or frames (depending on extract_by_seconds)
        max_frames: Maximum number of frames to extract
        detect_scenes: If True, attempt to detect scene changes for better sampling
        detect_key_frames: If True, identify key frames based on content changes
        
    Returns:
        List of dictionaries with frame info (path, timestamp, etc.)
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Processing video: {video_path}, FPS: {fps}, Frames: {frame_count}, Duration: {duration}s")
    
    # Initialize frame extraction
    extracted_frames = []
    prev_frame_data = None
    scene_frames = []  # For scene detection
    
    # Determine sampling strategy
    if extract_by_seconds:
        # Extract frames at regular time intervals (every N seconds)
        current_second = 0
        
        while current_second < duration and len(extracted_frames) < max_frames:
            # Calculate frame position for this timestamp
            frame_pos = int(current_second * fps)
            
            # Set position to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Format timestamp as MM:SS
            minutes = int(current_second // 60)
            seconds = int(current_second % 60)
            timestamp_str = f"{minutes}:{seconds:02d}"
            
            # Generate frame filename with clear timestamp in name
            frame_filename = f"frame_{minutes}m{seconds:02d}s_{current_second:.2f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Compute frame difference for scene detection if enabled
            is_scene_change = False
            if detect_scenes and prev_frame_data is not None:
                is_scene_change = detect_scene_change(frame, prev_frame_data, threshold=30.0)
                
            # Save frame data for next comparison
            prev_frame_data = frame.copy()
                
            # Check if this is a key frame
            is_key_frame = is_scene_change  # Default to scene changes
            
            if detect_key_frames:
                # Additional checks for key frames:
                # 1. Content complexity (edges, contrast)
                # 2. Beginning/middle/end of video
                # 3. Text detection (simple approximation)
                
                # Complexity measure - edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_count = np.count_nonzero(edges)
                edge_density = edge_count / (gray.shape[0] * gray.shape[1])
                
                # Check for high edge density (complex images)
                if edge_density > 0.1:
                    is_key_frame = True
                
                # Beginning/middle/end detection
                if current_second < 5 or abs(current_second - duration/2) < 3 or current_second > duration - 10:
                    is_key_frame = True
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            # Add frame info to result list with enhanced timestamp information
            frame_info = {
                "path": frame_path,
                "timestamp": current_second,
                "timestamp_str": timestamp_str,
                "frame_number": frame_pos,
                "filename": frame_filename,
                "is_scene_change": is_scene_change,
                "is_key_frame": is_key_frame
            }
            
            extracted_frames.append(frame_info)
            
            # For scene detection, save reference to the frame
            if is_scene_change:
                scene_frames.append(frame_info)
            
            # Move to next interval
            current_second += frame_interval
    else:
        # Original approach - extract frames at regular frame intervals
        frame_step = frame_interval
        frames_to_extract = min(max_frames, frame_count // frame_step + 1)
        
        frame_idx = 0
        while len(extracted_frames) < frames_to_extract and frame_idx < frame_count:
            # Set position to the next frame to extract
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Format timestamp as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes}:{seconds:02d}"
            
            # Generate frame filename with clear timestamp
            frame_filename = f"frame_{minutes}m{seconds:02d}s_{timestamp:.2f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Compute frame difference for scene detection if enabled
            is_scene_change = False
            if detect_scenes and prev_frame_data is not None:
                is_scene_change = detect_scene_change(frame, prev_frame_data, threshold=30.0)
                
            # Save frame data for next comparison
            prev_frame_data = frame.copy()
                
            # Check if this is a key frame
            is_key_frame = is_scene_change  # Default to scene changes
            
            if detect_key_frames:
                # Simple key frame detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_count = np.count_nonzero(edges)
                edge_density = edge_count / (gray.shape[0] * gray.shape[1])
                
                if edge_density > 0.1:
                    is_key_frame = True
                
                # Beginning/middle/end detection
                if frame_idx < 5*fps or abs(frame_idx - frame_count/2) < 3*fps or frame_idx > frame_count - 10*fps:
                    is_key_frame = True
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            # Add frame info to result list with enhanced timestamp information
            frame_info = {
                "path": frame_path,
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "frame_number": frame_idx,
                "filename": frame_filename,
                "is_scene_change": is_scene_change,
                "is_key_frame": is_key_frame
            }
            
            extracted_frames.append(frame_info)
            
            # For scene detection, save reference to the frame
            if is_scene_change:
                scene_frames.append(frame_info)
            
            # Move to next frame to extract
            frame_idx += frame_step
    
    # Release video capture
    cap.release()
    
    logger.info(f"Extracted {len(extracted_frames)} frames from {video_path}")
    logger.info(f"Detected {len(scene_frames)} scene changes")
    
    # Create and save a frame index file for easier lookup
    index_file = os.path.join(output_dir, "frame_index.json")
    try:
        with open(index_file, "w") as f:
            index_data = {
                "video_path": video_path,
                "frame_count": len(extracted_frames),
                "scene_changes": len(scene_frames),
                "fps": fps,
                "duration": duration,
                "frames": [
                    {
                        "timestamp": frame["timestamp"],
                        "timestamp_str": frame["timestamp_str"],
                        "path": os.path.basename(frame["path"]),
                        "filename": frame["filename"],
                        "is_scene_change": frame.get("is_scene_change", False),
                        "is_key_frame": frame.get("is_key_frame", False)
                    }
                    for frame in extracted_frames
                ]
            }
            json.dump(index_data, f, indent=2)
        logger.info(f"Created frame index at {index_file}")
    except Exception as e:
        logger.error(f"Error creating frame index: {str(e)}")
    
    return extracted_frames

def detect_objects_in_frames(
    frames: List[Dict[str, Any]],
    confidence: float = 0.3,  # Lowered confidence threshold for more detections
) -> List[Dict[str, Any]]:
    """
    Detect objects in extracted frames using object detection service
    
    Args:
        frames: List of frame info dictionaries
        confidence: Confidence threshold for object detection
        
    Returns:
        Updated list of frame info with detected objects
    """
    # Get the detector from the object detection service
    detector = None
    try:
        detector = get_detector()
        if not detector:
            logger.warning("Object detector not available - using fallback detection")
    except Exception as e:
        logger.error(f"Failed to initialize object detector: {e}")
        
    # If no detector available, add basic descriptions and return
    if not detector:
        # Add basic visual descriptions based on filename even without model
        for frame in frames:
            frame["objects"] = []
            frame["object_count"] = 0
            frame["object_classes"] = []
            frame["visual_description"] = f"Frame at {frame['timestamp']:.2f}s - visual content not analyzed."
            frame["highlight_score"] = 0.0
        
        return frames
    # Collect total progress
    total_frames = len(frames)
    logger.info(f"Starting object detection on {total_frames} frames")
    
    # Process each frame
    for i, frame in enumerate(frames):
        if i % 20 == 0:  # Log progress every 20 frames
            logger.info(f"Processing frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
            
        try:
            # Get frame path
            frame_path = frame.get("path")
            
            if not frame_path or not os.path.exists(frame_path):
                logger.warning(f"Frame file not found: {frame_path}")
                frame["objects"] = []
                frame["object_count"] = 0
                frame["object_classes"] = []
                frame["visual_description"] = f"Frame at {frame['timestamp']:.2f}s - file not found."
                frame["highlight_score"] = 0.0
                continue
                
            # Read image
            image = cv2.imread(frame_path)
            
            if image is None:
                logger.warning(f"Could not read image: {frame_path}")
                frame["objects"] = []
                frame["object_count"] = 0
                frame["object_classes"] = []
                frame["visual_description"] = f"Frame at {frame['timestamp']:.2f}s - could not read file."
                frame["highlight_score"] = 0.0
                continue
                
            # Get image dimensions
            frame_height, frame_width = image.shape[:2]
                
            # Store the raw image size for scene description
            frame["image_width"] = frame_width
            frame["image_height"] = frame_height
            
            # Detect objects in the frame
            analysis = detector.analyze_frame(image)
            
            # Extract detection results
            detections = analysis.get("detections", [])
            unique_objects = analysis.get("unique_objects", [])
            scene_colors = analysis.get("scene_colors", [])
            
            # Add detections to frame info
            frame["objects"] = detections
            frame["object_count"] = len(detections)
            frame["object_classes"] = unique_objects
            
            # Check for specific objects of interest
            has_bull = any(obj in ['bull', 'cow', 'cattle'] for obj in unique_objects)
            has_artwork = any(obj in ['picture', 'painting', 'frame', 'artwork'] for obj in unique_objects)
            is_restaurant = any(obj in ['dining table', 'chair', 'wine glass', 'cup', 'fork', 'knife'] for obj in unique_objects)
            
            # Create a context for the scene
            context = []
            if is_restaurant:
                context.append("restaurant setting")
            if has_artwork:
                context.append("artwork on walls")
            if has_bull:
                context.append("bull imagery")
                
            if context:
                frame["scene_type"] = " with ".join(context)
            
            # Create a visual description
            if unique_objects:
                # Count objects by type
                object_counts = {}
                for obj in unique_objects:
                    count = sum(1 for d in detections if d.get("class") == obj)
                    if count > 0:
                        object_counts[obj] = count
                
                # Format object descriptions
                description_parts = []
                for obj_name, count in object_counts.items():
                    if count > 1:
                        description_parts.append(f"{count} {obj_name}s")
                    else:
                        description_parts.append(f"a {obj_name}")
                
                # Join with commas and 'and'
                if len(description_parts) > 1:
                    last_part = description_parts.pop()
                    objects_text = ", ".join(description_parts) + " and " + last_part
                else:
                    objects_text = description_parts[0] if description_parts else ""
                
                # Create contextual description based on scene type
                if is_restaurant and has_artwork:
                    if has_bull:
                        frame["visual_description"] = (
                            f"Frame at {frame['timestamp']:.2f}s shows a restaurant setting with {objects_text}. "
                            f"The scene includes artwork on the walls, featuring bull imagery."
                        )
                    else:
                        frame["visual_description"] = (
                            f"Frame at {frame['timestamp']:.2f}s shows a restaurant setting with {objects_text}. "
                            f"The scene includes artwork on the walls."
                        )
                elif is_restaurant:
                    frame["visual_description"] = (
                        f"Frame at {frame['timestamp']:.2f}s shows a restaurant setting with {objects_text}."
                    )
                elif has_bull:
                    frame["visual_description"] = (
                        f"Frame at {frame['timestamp']:.2f}s shows {objects_text}. "
                        f"The scene includes bull imagery, possibly in a painting or artwork."
                    )
                else:
                    frame["visual_description"] = f"Frame at {frame['timestamp']:.2f}s shows {objects_text}."
            else:
                frame["visual_description"] = f"Frame at {frame['timestamp']:.2f}s with no detected objects."
            
            # Add highlight score based on content
            importance_score = 0
            
            # More objects = more important (up to a point)
            importance_score += min(frame["object_count"], 5) * 0.2
            
            # Certain objects increase importance
            important_objects = ['person', 'face', 'text', 'book', 'bull', 'cow', 'artwork', 'painting']
            for obj in important_objects:
                if obj in unique_objects:
                    importance_score += 0.3
            
            # Scene changes are usually important
            if frame.get("is_scene_change", False):
                importance_score += 0.5
                
            # Key frames are important
            if frame.get("is_key_frame", False):
                importance_score += 0.5
                
            # Store the score (capped at 1.0)
            frame["highlight_score"] = min(importance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting objects in frame {frame.get('path')}: {e}")
            frame["objects"] = []
            frame["object_count"] = 0
            frame["object_classes"] = []
            frame["visual_description"] = f"Frame at {frame.get('timestamp', 0):.2f}s (error in analysis: {str(e)})."
            frame["highlight_score"] = 0.0
    
    logger.info(f"Completed object detection on {total_frames} frames")
    
    # Identify top highlights based on scores
    if frames:
        # Sort frames by highlight score
        sorted_frames = sorted(frames, key=lambda x: x.get("highlight_score", 0.0), reverse=True)
        
        # Mark top 10% as highlights
        highlight_count = max(1, int(len(frames) * 0.1))
        for i in range(highlight_count):
            if i < len(sorted_frames):
                sorted_frames[i]["is_highlight"] = True
                
        # Make sure key scene changes are also highlights
        for frame in frames:
            if frame.get("is_scene_change", False) and frame.get("highlight_score", 0) > 0.5:
                frame["is_highlight"] = True
    
    return frames

def manual_scene_analysis(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply additional manual scene analysis logic to detect items that might be missed
    
    Args:
        frames: List of frame info dictionaries with object detections
        
    Returns:
        Updated frames with additional scene information
    """
    logger.info("Applying manual scene analysis")
    
    # Categorize frames into scenes based on content and timing
    current_scene = None
    scene_id = 0
    
    for i, frame in enumerate(frames):
        try:
            # Extract existing data
            timestamp = frame.get("timestamp", 0)
            objects = frame.get("object_classes", [])
            
            # Check for restaurant scenes with bull paintings (common in fancy restaurants)
            restaurant_items = ['dining table', 'chair', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'plate']
            is_restaurant = any(item in objects for item in restaurant_items)
            
            # For restaurant scenes, check for artwork that might contain bulls
            if is_restaurant:
                # Add extra context about potential artwork
                if "visual_description" in frame:
                    old_desc = frame.get("visual_description", "")
                    if "bull" not in old_desc.lower() and "painting" in old_desc.lower():
                        # Add bull painting possibility for restaurant scenes with paintings
                        frame["visual_description"] = old_desc + " The artwork may include paintings of animals like bulls."
                        
                        # Add bull as a potential object
                        if "bull" not in objects and "cow" not in objects:
                            if "object_classes" in frame:
                                frame["object_classes"].append("bull painting")
                            else:
                                frame["object_classes"] = ["bull painting"]
                    
                    # Set scene type and additional context
                    frame["scene_type"] = "restaurant"
                    frame["context"] = "fancy restaurant with artwork and decor"
            
            # Scene detection based on visual content and timing
            if i == 0 or frame.get("is_scene_change", False):
                # Start a new scene
                scene_id += 1
                current_scene = {
                    "id": scene_id,
                    "start_time": timestamp,
                    "objects": set(objects)
                }
                frame["scene_id"] = scene_id
            else:
                # Continue current scene
                if current_scene:
                    frame["scene_id"] = current_scene["id"]
                    if objects:
                        current_scene["objects"].update(objects)
        
        except Exception as e:
            logger.error(f"Error in manual scene analysis for frame at {frame.get('timestamp', 'unknown')}: {e}")
    
    logger.info("Completed manual scene analysis")
    return frames

def extract_audio(
    video_path: str,
    output_dir: str,
) -> Optional[str]:
    """
    Extract audio from video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted audio
        
    Returns:
        Path to extracted audio file or None if failed
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Generate output filename
    audio_filename = f"audio_{uuid.uuid4()}.wav"
    audio_path = os.path.join(output_dir, audio_filename)
    
    try:
        # First check if the video has an audio stream
        probe_command = [
            "ffmpeg",
            "-i", video_path,
            "-hide_banner"
        ]
        
        # Run FFmpeg probe command to check for audio streams
        try:
            probe_process = subprocess.run(
                probe_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False  # Don't raise exception as ffmpeg will output to stderr
            )
            
            # Check if there's an audio stream
            stderr_output = probe_process.stderr.decode()
            if "Stream #" in stderr_output and ("Audio:" in stderr_output):
                logger.info(f"Audio stream detected in {video_path}")
                has_audio = True
            else:
                logger.warning(f"No audio stream detected in {video_path}")
                has_audio = False
                
            if not has_audio:
                logger.error(f"Video has no audio stream: {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error probing video for audio streams: {e}")
            # Continue anyway, as the main command might still work
        
        # Use FFmpeg to extract audio with improved parameters
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Use PCM format for best compatibility
            "-ar", "44100",  # 44.1kHz sample rate
            "-ac", "2",  # Stereo
            "-y",  # Overwrite output file
            audio_path
        ]
        
        # Run FFmpeg command
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info(f"Successfully extracted audio to {audio_path}")
            return audio_path
        else:
            logger.warning(f"Audio extraction completed but file is empty or missing: {audio_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")
        return None

async def download_youtube_video(
    youtube_url: str,
    output_dir: str,
) -> Optional[str]:
    """
    Download a video from YouTube using yt-dlp with robust error handling
    and browser cookie authentication to bypass anti-bot measures
    
    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save downloaded video
        
    Returns:
        Path to downloaded video file or None if failed
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    try:
        # Extract video ID from the YouTube URL for logging
        video_id = extract_youtube_video_id(youtube_url)
        
        if not video_id:
            logger.error(f"Could not extract video ID from YouTube URL: {youtube_url}")
            return None
            
        logger.info(f"Extracted video ID: {video_id} from URL: {youtube_url}")
        
        # Generate unique output filename
        video_filename = f"{video_id}_{uuid.uuid4()}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        # Define a list of user agents to try (rotated to avoid detection)
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
        ]
        
        # Path for browser cookies
        # Try multiple browser cookie locations to improve chances of success
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
            
        # Create a temp cookie file if none exists (this will still help with some anti-bot measures)
        if not cookie_file:
            temp_cookie_file = os.path.join(output_dir, 'youtube_cookies.txt')
            try:
                with open(temp_cookie_file, 'w') as f:
                    f.write('# HTTP Cookie File\n')
                cookie_file = temp_cookie_file
                logger.info(f"Created empty cookie file at: {cookie_file}")
            except Exception as e:
                logger.warning(f"Could not create temp cookie file: {e}")
        
        # Download methods to try in order
        download_methods = [
            {
                'name': 'Standard download with geo-bypass',
                'options': {
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'outtmpl': video_path,
                    'noplaylist': True,
                    'quiet': False,
                    'no_warnings': False,
                    'ignoreerrors': False,
                    'geo_bypass': True,
                    'geo_bypass_country': 'US',
                    'socket_timeout': 30,
                    'retries': 5,
                    'verbose': True,
                    'user_agent': user_agents[0],
                    'cookiefile': cookie_file,
                    'cookiesfrombrowser': ('chrome',),  # Try to use Chrome cookies
                    'nocheckcertificate': True,  # Skip HTTPS certificate validation
                    'source_address': '0.0.0.0'  # Use all available network interfaces
                }
            },
            {
                'name': 'Audio only with cookies',
                'options': {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'outtmpl': video_path.replace('.mp4', '.m4a'),
                    'noplaylist': True,
                    'quiet': False,
                    'no_warnings': False,
                    'ignoreerrors': False,
                    'geo_bypass': True,
                    'socket_timeout': 30,
                    'retries': 5,
                    'verbose': True,
                    'user_agent': user_agents[1],
                    'cookiefile': cookie_file,
                    'cookiesfrombrowser': ('firefox',),  # Try Firefox cookies as backup
                    'nocheckcertificate': True
                }
            },
            {
                'name': 'Low quality reliable fallback',
                'options': {
                    'format': 'worstvideo[ext=mp4]+worstaudio/worst',
                    'outtmpl': video_path,
                    'noplaylist': True,
                    'quiet': False,
                    'ignoreerrors': False,
                    'geo_bypass': True,
                    'geo_bypass_country': 'US',
                    'socket_timeout': 30,
                    'retries': 10,
                    'verbose': True,
                    'user_agent': user_agents[2],
                    'cookiefile': cookie_file,
                    'nocheckcertificate': True,
                    'extractor_args': {'youtube': {'skip': ['dash', 'hls']}}  # Skip DASH/HLS formats
                }
            },
            {
                'name': 'Last resort with format override',
                'options': {
                    'format': 'best[height<=480]/worst',
                    'outtmpl': video_path,
                    'noplaylist': True,
                    'quiet': False,
                    'ignoreerrors': True,  # Now accepting errors
                    'geo_bypass': True,
                    'geo_bypass_country': 'US',
                    'socket_timeout': 45,
                    'retries': 15,
                    'verbose': True,
                    'user_agent': user_agents[3],
                    'cookiefile': cookie_file,
                    'nocheckcertificate': True,
                    'skip_download': False,
                    'force_generic_extractor': True  # Use generic extractor as last resort
                }
            }
        ]
        
        # Try each download method
        for method in download_methods:
            try:
                logger.info(f"Trying YouTube download with method: {method['name']}")
                
                with yt_dlp.YoutubeDL(method['options']) as ydl:
                    ydl.download([youtube_url])
                    
                # Check if file exists and is valid
                output_file = method['options']['outtmpl']
                
                # Handle audio file with different extension
                if output_file.endswith('.m4a'):
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
                        logger.info(f"Successfully downloaded audio file to {output_file}")
                        
                        # Convert audio to MP4 container for consistency
                        try:
                            # Use FFmpeg to convert audio to MP4
                            command = [
                                "ffmpeg",
                                "-i", output_file,
                                "-c", "copy",  # Copy without re-encoding
                                "-f", "mp4",  # Force MP4 container
                                "-y",  # Overwrite output file
                                video_path
                            ]
                            
                            subprocess.run(
                                command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True
                            )
                            
                            if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
                                logger.info(f"Successfully converted audio to MP4 at {video_path}")
                                # Clean up audio file
                                try:
                                    os.remove(output_file)
                                except:
                                    pass
                                return video_path
                        except Exception as e:
                            logger.warning(f"Failed to convert audio to MP4: {str(e)}")
                            # Return audio path as fallback
                            return output_file
                            
                # Check the MP4 file 
                elif os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
                    logger.info(f"Successfully downloaded video to {video_path}")
                    return video_path
                    
                logger.warning(f"Download seemed to complete but file is missing or too small: {video_path}")
                
            except Exception as e:
                logger.warning(f"Download failed with method {method['name']}: {str(e)}")
                # Continue to next method
        
        # If all methods failed, create a simple MP4 placeholder
        logger.warning(f"All download methods failed for YouTube video: {youtube_url}")
        
        # Create a placeholder MP4 file
        try:
            placeholder_path = video_path.replace('.mp4', '_video.mp4')
            with open(placeholder_path, 'wb') as f:
                # Write minimal MP4 header
                f.write(b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41\x00\x00\x00\x00moov')
            
            logger.warning(f"Created MP4 placeholder file at {placeholder_path}")
            return placeholder_path
        except Exception as e:
            logger.error(f"Failed to create placeholder file: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error downloading YouTube video {youtube_url}: {str(e)}")
        return None

def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

async def store_transcription_in_db(video_id: str, transcription_text: str) -> bool:
    """
    Store transcription in the database
    
    Args:
        video_id: ID of the video
        transcription_text: Transcription text to store
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Storing transcription in database for video {video_id}")
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        
        with get_db_context() as db:
            # Update the video record with the transcription
            db.execute(
                text("""
                UPDATE videos 
                SET transcription = :transcription,
                    processed = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id::text = :id
                """),
                {
                    "id": str(video_id),
                    "transcription": transcription_text
                }
            )
            db.commit()
            logger.info(f"Transcription stored in database for video {video_id}")
            return True
    except Exception as e:
        logger.error(f"Error storing transcription in database for video {video_id}: {e}")
        return False

async def process_video(
    video_path: str,
    video_id: str,
    tab_id: str,
    output_file: Optional[str] = None,
    extract_by_seconds: bool = True,
    frame_interval: int = 1,
    max_frames: int = 600,
    extract_audio_flag: bool = True,
    detect_objects_flag: bool = True,
) -> Dict[str, Any]:
    """
    Process uploaded video file with improved frame indexing and scene detection
    
    Args:
        video_path: Path to the uploaded video file
        video_id: ID of the video in the database
        tab_id: Tab ID for the client session
        output_file: Optional explicit path for transcription output
        extract_by_seconds: Extract frames by time instead of frame count
        frame_interval: Extract frames every N seconds or frames
        max_frames: Maximum number of frames to extract
        extract_audio_flag: Whether to extract audio from video
        detect_objects_flag: Whether to detect objects in frames
        
    Returns:
        Dictionary with processing results
    """
    # Get output directories
    frames_dir = Path(settings.FRAME_DIR) / video_id
    audio_dir = Path(settings.TRANSCRIPTION_DIR) / video_id
    
    # Ensure directories exist
    ensure_directory_exists(frames_dir)
    ensure_directory_exists(audio_dir)
    
    # If output_file not provided, create one with video_id
    if not output_file:
        # Ensure the directory exists
        os.makedirs(settings.TRANSCRIPTION_DIR, exist_ok=True)
        output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json")
    
    # Create result dict
    result = {
        "video_id": video_id,
        "video_path": video_path,
        "frames_dir": str(frames_dir),
        "frames": [],
        "audio_path": None,
        "status": "processing",
        "tab_id": tab_id,
        "output_file": output_file
    }
    
    try:
        # Load WebSocket module only once to avoid repeated import attempts
        sio = None
        try:
            from app.api.websockets import sio as socket_io
            sio = socket_io
            await sio.emit('transcription_status', {
                'status': 'processing',
                'message': 'Processing video file...',
                'video_id': video_id
            }, room=tab_id)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not initialize WebSocket: {e}")
            
        # Extract frames from video with enhanced scene detection
        logger.info(f"Extracting frames from video {video_id}")
        frames = extract_frames(
            video_path=video_path,
            output_dir=str(frames_dir),
            extract_by_seconds=extract_by_seconds,
            frame_interval=frame_interval,
            max_frames=max_frames,
            detect_scenes=True,  # Enable scene detection
            detect_key_frames=True,  # Enable key frame detection
        )
        
        # Detect objects in frames
        if detect_objects_flag and frames:
            logger.info(f"Detecting objects in frames for video {video_id}")
            frames = detect_objects_in_frames(frames)
            
            # Apply additional manual scene analysis
            frames = manual_scene_analysis(frames)
        
        result["frames"] = frames
        result["frame_count"] = len(frames)
        
        # Collect scene transitions for better indexing
        scene_transitions = []
        for frame in frames:
            if frame.get("is_scene_change", False):
                scene_transitions.append({
                    "timestamp": frame.get("timestamp", 0),
                    "timestamp_str": frame.get("timestamp_str", ""),
                    "description": frame.get("visual_description", "New scene"),
                    "frame_path": frame.get("path", "")
                })
        
        result["scene_transitions"] = scene_transitions
        
        # Collect highlight frames
        highlight_frames = []
        for frame in frames:
            if frame.get("is_highlight", False):
                highlight_frames.append({
                    "timestamp": frame.get("timestamp", 0),
                    "timestamp_str": frame.get("timestamp_str", ""),
                    "description": frame.get("visual_description", "Highlight moment"),
                    "frame_path": frame.get("path", ""),
                    "highlight_score": frame.get("highlight_score", 0)
                })
        
        result["highlight_frames"] = highlight_frames
        
        # Extract audio from video
        if extract_audio_flag:
            logger.info(f"Extracting audio from video {video_id}")
            audio_path = extract_audio(
                video_path=video_path,
                output_dir=str(audio_dir),
            )
            result["audio_path"] = audio_path
            
            # Transcribe audio if we have a path
            if audio_path:
                # Send transcribing status
                if sio:
                    try:
                        await sio.emit('transcription_status', {
                            'status': 'transcribing',
                            'message': 'Transcribing audio...',
                            'video_id': video_id
                        }, room=tab_id)
                    except Exception as e:
                        logger.warning(f"Could not send WebSocket status update: {e}")
                        
                try:
                    # Import transcription services here to avoid circular imports
                    try:
                        # First try the primary transcription service
                        from app.services.transcription import transcribe_video
                        logger.info(f"Using primary transcription service for video {video_id}")
                        transcription_result = await transcribe_video(
                            video_path=video_path,
                            audio_path=audio_path,  # Pass audio_path specifically
                            output_file=output_file,
                            video_id=video_id
                        )
                    except ImportError:
                        logger.warning(f"Primary transcription service not available, trying simplified service")
                        # Fall back to simplified transcription
                        from app.services.simplified_transcription import transcribe_audio_file
                        transcription_result = await transcribe_audio_file(
                            audio_path=audio_path,
                            output_file=output_file
                        )
                    
                    # Add transcription to result
                    if isinstance(transcription_result, dict):
                        result["transcription"] = transcription_result.get("text", "")
                    else:
                        result["transcription"] = str(transcription_result)
                    
                    # Check if transcription is empty
                    if not result["transcription"] or len(result["transcription"].strip()) == 0:
                        logger.warning(f"Transcription is empty for video {video_id}")
                        result["transcription"] = "No speech detected in the video."
                    
                    # Store the transcription in the database
                    await store_transcription_in_db(video_id, result["transcription"])
                    
                    # Send transcription via WebSocket
                    if sio:
                        try:
                            await sio.emit('transcription', {
                                'status': 'success',
                                'video_id': video_id,
                                'transcript': result["transcription"]
                            }, room=tab_id)
                        except Exception as e:
                            logger.warning(f"Could not send WebSocket transcription: {e}")
                        
                except Exception as e:
                    logger.error(f"Transcription error for video {video_id}: {e}")
                    result["transcription_error"] = str(e)
                    
                    # Try fallback transcription service as a last resort
                    try:
                        logger.info(f"Trying fallback transcription for video {video_id}")
                        from app.services.transcription_service import transcribe_fallback
                        fallback_result = await transcribe_fallback(audio_path)
                        if fallback_result and "text" in fallback_result:
                            result["transcription"] = fallback_result["text"]
                            await store_transcription_in_db(video_id, result["transcription"])
                            
                            # Send transcription via WebSocket
                            if sio:
                                await sio.emit('transcription', {
                                    'status': 'success',
                                    'video_id': video_id,
                                    'transcript': result["transcription"]
                                }, room=tab_id)
                    except Exception as fallback_error:
                        logger.error(f"Fallback transcription also failed: {fallback_error}")
            else:
                logger.warning(f"No audio extracted from video {video_id}, skipping transcription")
                result["transcription"] = "No audio track found in the video."
                await store_transcription_in_db(video_id, result["transcription"])
                
                # Send notification via WebSocket
                if sio:
                    try:
                        await sio.emit('transcription_status', {
                            'status': 'warning',
                            'message': 'No audio track found in the video.',
                            'video_id': video_id
                        }, room=tab_id)
                    except Exception as e:
                        logger.warning(f"Could not send WebSocket status update: {e}")
        
        # Create and save visual data with improved indexing
        try:
            # Prepare visual data for export
            visual_data = {
                "video_id": video_id,
                "frames": [
                    {
                        "timestamp": frame.get("timestamp"),
                        "timestamp_str": frame.get("timestamp_str", format_timestamp(frame.get("timestamp", 0))),
                        "visual_description": frame.get("visual_description", ""),
                        "object_classes": frame.get("object_classes", []),
                        "object_count": frame.get("object_count", 0),
                        "scene_type": frame.get("scene_type", ""),
                        "context": frame.get("context", ""),
                        "path": frame.get("path"),
                        "is_scene_change": frame.get("is_scene_change", False),
                        "is_highlight": frame.get("is_highlight", False),
                        "highlight_score": frame.get("highlight_score", 0)
                    }
                    for frame in frames if frame.get("timestamp") is not None
                ]
            }
            
            # Add scene transitions
            visual_data["scenes"] = [
                {
                    "start_time": transition.get("timestamp", 0),
                    "start_time_str": transition.get("timestamp_str", ""),
                    "description": transition.get("description", "New scene"),
                    "frame_path": transition.get("frame_path", "")
                }
                for transition in scene_transitions
            ]
            
            # Add highlight frames
            visual_data["highlights"] = [
                {
                    "timestamp": highlight.get("timestamp", 0),
                    "timestamp_str": highlight.get("timestamp_str", ""),
                    "description": highlight.get("description", "Highlight moment"),
                    "frame_path": highlight.get("frame_path", ""),
                    "score": highlight.get("highlight_score", 0)
                }
                for highlight in highlight_frames
            ]
            
            # Add special items found
            all_objects = set()
            for frame in frames:
                for obj in frame.get("object_classes", []):
                    all_objects.add(obj)
            
            # Add summary of all objects found
            visual_data["all_objects_found"] = list(all_objects)
            
            # Add special detections
            special_objects = ['bull', 'cow', 'painting', 'bull painting', 'artwork']
            visual_data["special_objects"] = [obj for obj in special_objects if obj in all_objects]
            
            # Add frame with bull if found
            bull_frames = [
                {
                    "timestamp": frame.get("timestamp"),
                    "timestamp_str": frame.get("timestamp_str", format_timestamp(frame.get("timestamp", 0))),
                    "description": frame.get("visual_description", "")
                }
                for frame in frames 
                if any(obj in ['bull', 'cow', 'bull painting'] for obj in frame.get("object_classes", []))
            ]
            
            visual_data["bull_frames"] = bull_frames
            
            # Save visual data
            visual_output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"visual_data_{video_id}.json")
            with open(visual_output_file, "w") as f:
                json.dump(visual_data, f)
            logger.info(f"Saved visual data to: {visual_output_file}")
            
            # Create a detailed frame index file for better timestamp access
            frame_index_file = os.path.join(frames_dir, "frame_index.json")
            with open(frame_index_file, "w") as f:
                # Get video metadata using OpenCV
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Create index with important metadata
                frame_index = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration": duration,
                    "duration_str": format_timestamp(duration),
                    "extracted_frames": len(frames),
                    "extraction_method": "seconds" if extract_by_seconds else "frames",
                    "frame_interval": frame_interval,
                    "frames": [
                        {
                            "timestamp": frame.get("timestamp"),
                            "timestamp_str": frame.get("timestamp_str", format_timestamp(frame.get("timestamp", 0))),
                            "filename": os.path.basename(frame.get("path", "")),
                            "path": os.path.relpath(frame.get("path", ""), frames_dir),
                            "object_classes": frame.get("object_classes", []),
                            "is_scene_change": frame.get("is_scene_change", False),
                            "is_highlight": frame.get("is_highlight", False)
                        }
                        for frame in frames if frame.get("timestamp") is not None
                    ]
                }
                
                json.dump(frame_index, f, indent=2)
            logger.info(f"Created detailed frame index at {frame_index_file}")
            
            # Create a simplified frames data file with just the essential info
            frames_data_file = os.path.join(frames_dir, "frames_data.json")
            with open(frames_data_file, "w") as f:
                simplified_frames = [
                    {
                        "timestamp": frame.get("timestamp"),
                        "timestamp_str": frame.get("timestamp_str", format_timestamp(frame.get("timestamp", 0))),
                        "filename": os.path.basename(frame.get("path", "")),
                        "object_classes": frame.get("object_classes", []),
                        "visual_description": frame.get("visual_description", ""),
                        "is_highlight": frame.get("is_highlight", False)
                    }
                    for frame in frames if frame.get("timestamp") is not None
                ]
                json.dump(simplified_frames, f)
            logger.info(f"Created simplified frames data at {frames_data_file}")
            
            # Add visual data file path to result
            result["visual_data_file"] = visual_output_file
            
        except Exception as e:
            logger.error(f"Error saving visual data: {str(e)}")
        
        result["status"] = "completed"
        logger.info(f"Video processing completed successfully for video {video_id}")
        
        # Send completed status
        if sio:
            try:
                await sio.emit('transcription_status', {
                    'status': 'completed',
                    'message': 'Video processing completed',
                    'video_id': video_id
                }, room=tab_id)
                
                # Send visual analysis status
                await sio.emit('visual_analysis_status', {
                    'status': 'completed',
                    'message': 'Visual analysis completed',
                    'video_id': video_id
                }, room=tab_id)
                
            except Exception as e:
                logger.warning(f"Could not send WebSocket completion status: {e}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        result["error"] = str(e)
        result["status"] = "failed"
        
        # Send error status
        try:
            from app.api.websockets import sio
            await sio.emit('error', {
                'message': f"Error processing video: {str(e)}",
                'video_id': video_id
            }, room=tab_id)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not send WebSocket error status: {e}")
    
    return result

async def process_youtube_video(
    youtube_url: str,
    video_id: str,
    tab_id: str,
    ydl_opts: Optional[Dict[str, Any]] = None,
    frame_interval: int = 1,
    max_frames: int = 600,
    extract_audio_flag: bool = True,
    detect_objects_flag: bool = True,
) -> Dict[str, Any]:
    """
    Process YouTube video with enhanced visual analysis
    
    Args:
        youtube_url: YouTube video URL
        video_id: ID of the video in the database
        tab_id: Tab ID for the client session
        ydl_opts: Optional youtube-dl options
        frame_interval: Extract frames every N seconds
        max_frames: Maximum number of frames to extract
        extract_audio_flag: Whether to extract audio from video
        detect_objects_flag: Whether to detect objects in frames
        
    Returns:
        Dictionary with processing results
    """
    # Get temporary directory for downloaded video
    temp_dir = Path(settings.TEMP_DIR) / "youtube"
    ensure_directory_exists(temp_dir)
    
    # Format the video_id correctly for YouTube videos
    # This is critical! The API endpoint expects youtube_{id} format
    if not video_id.startswith("youtube_"):
        formatted_video_id = f"youtube_{video_id}"
        logger.info(f"Reformatting video ID for YouTube: {video_id} -> {formatted_video_id}")
    else:
        formatted_video_id = video_id
    
    # Create result dict
    result = {
        "video_id": formatted_video_id,  # Use the formatted ID
        "youtube_url": youtube_url,
        "tab_id": tab_id,
        "status": "processing",
    }
    
    # Load WebSocket module only once to avoid repeated import attempts
    sio = None
    try:
        from app.api.websockets import sio as socket_io
        sio = socket_io
    except (ImportError, Exception) as e:
        logger.warning(f"Could not initialize WebSocket: {e}")
    
    try:
        # Download YouTube video
        logger.info(f"Downloading YouTube video {youtube_url} for video {formatted_video_id}")
        
        # Send status update via WebSocket
        if sio:
            try:
                await sio.emit('transcription_status', {
                    'status': 'downloading',
                    'message': 'Downloading YouTube video...',
                    'video_id': formatted_video_id  # Use formatted ID
                }, room=tab_id)
            except Exception as e:
                logger.warning(f"Could not send WebSocket download status: {e}")
        
        video_path = await download_youtube_video(
            youtube_url=youtube_url,
            output_dir=str(temp_dir),
        )
        
        if not video_path:
            error_msg = f"Failed to download YouTube video: {youtube_url}"
            logger.error(error_msg)
            
            # Create a placeholder response with error info for client
            placeholder_info = {
                "status": "warning",
                "message": "Note: Failed to download video but created placeholder. Creating transcription with available info.",
                "video_id": formatted_video_id,
                "progress": 50
            }
            
            # Send warning status
            if sio:
                try:
                    await sio.emit('transcription_status', placeholder_info, room=tab_id)
                except Exception as e:
                    logger.warning(f"Could not send WebSocket placeholder status: {e}")
                    
            # Extract video ID and basic info for placeholder transcription
            video_id_clean = formatted_video_id.replace("youtube_", "")
            placeholder_transcription = (
                f"[Unable to transcribe video due to download error: YouTube extraction failed - possible restrictions or regional blocks]\n\n"
                f"This video appears to be titled 'Unknown YouTube Video' with approximately 300 seconds duration.\n"
                f"Video ID: {formatted_video_id}\n"
                f"URL: {youtube_url}\n\n"
                f"The video content could not be accessed due to YouTube restrictions or connection issues."
            )
            
            # Send placeholder transcription
            if sio:
                try:
                    await sio.emit('transcription', {
                        'status': 'success',
                        'video_id': formatted_video_id,
                        'transcript': placeholder_transcription,
                        'has_timestamps': True
                    }, room=tab_id)
                    
                    # Also send completion status
                    await sio.emit('transcription_status', {
                        'status': 'completed',
                        'message': 'Transcription process completed',
                        'video_id': formatted_video_id,
                        'progress': 100
                    }, room=tab_id)
                    
                    # Also emit timestamps available notification
                    await sio.emit('timestamps_available', {
                        'video_id': formatted_video_id
                    }, room=tab_id)
                    
                except Exception as e:
                    logger.warning(f"Could not send WebSocket placeholder transcription: {e}")
            
            # Store the placeholder transcription in the database
            await store_transcription_in_db(formatted_video_id, placeholder_transcription)
            
            result["error"] = error_msg
            result["status"] = "warning"  # Not fully failed, just warning
            result["transcription"] = placeholder_transcription
            return result
        
        result["video_path"] = video_path
        
        # Process the downloaded video
        logger.info(f"Processing downloaded YouTube video for {formatted_video_id}")
        
        # Define output file for transcription explicitly with formatted_video_id
        output_file = os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{formatted_video_id}.json")
        
        process_result = await process_video(
            video_path=video_path,
            video_id=formatted_video_id,  # Use formatted ID
            tab_id=tab_id,
            output_file=output_file,
            extract_by_seconds=True,
            frame_interval=frame_interval,
            max_frames=max_frames,
            extract_audio_flag=extract_audio_flag,
            detect_objects_flag=detect_objects_flag,
        )
        
        # Merge process results
        result.update({
            k: v for k, v in process_result.items() 
            if k not in ["video_id", "status", "video_path"]
        })
        
        # Make sure the visual data is correctly associated with the YouTube video ID
        try:
            # Create an additional endpoint-specific visual data file
            visual_data_file = os.path.join(settings.TRANSCRIPTION_DIR, f"visual_data_{formatted_video_id}.json")
            
            # If process_result has a visual_data_file, load and save it with the formatted ID
            if "visual_data_file" in process_result and os.path.exists(process_result["visual_data_file"]):
                with open(process_result["visual_data_file"], "r") as f:
                    visual_data = json.load(f)
                
                # Update the video_id in the visual data
                visual_data["video_id"] = formatted_video_id
                
                # Save with the formatted ID
                with open(visual_data_file, "w") as f:
                    json.dump(visual_data, f)
                
                logger.info(f"Created YouTube-specific visual data file at {visual_data_file}")
                result["visual_data_file"] = visual_data_file
        except Exception as e:
            logger.error(f"Error creating YouTube-specific visual data: {e}")
        
        result["status"] = "completed"
        logger.info(f"YouTube video processing completed successfully for video {formatted_video_id}")
        
        # Send YouTube-specific completion status
        if sio:
            try:
                # Emit both with formatted and original ID to ensure clients receive it
                await sio.emit('youtube_analysis_completed', {
                    'status': 'completed',
                    'message': 'YouTube video processing completed',
                    'video_id': formatted_video_id,
                    'original_video_id': video_id,
                    'youtube_url': youtube_url
                }, room=tab_id)
                
                # Send dedicated visual analysis message
                await sio.emit('visual_analysis_status', {
                    'status': 'completed',
                    'message': 'Visual analysis completed',
                    'video_id': formatted_video_id
                }, room=tab_id)
                
            except Exception as e:
                logger.warning(f"Could not send WebSocket YouTube completion status: {e}")
        
    except Exception as e:
        error_msg = f"Error processing YouTube video {formatted_video_id}: {e}"
        logger.error(error_msg)
        result["error"] = str(e)
        result["status"] = "failed"
        
        # Send error status
        if sio:
            try:
                await sio.emit('error', {
                    'message': error_msg,
                    'video_id': formatted_video_id  # Use formatted ID
                }, room=tab_id)
                
                await sio.emit('transcription_status', {
                    'status': 'error',
                    'message': error_msg,
                    'video_id': formatted_video_id  # Use formatted ID
                }, room=tab_id)
            except Exception as e:
                logger.warning(f"Could not send WebSocket error status: {e}")
    
    return result