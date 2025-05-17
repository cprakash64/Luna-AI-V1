import cv2
import torch
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetector:
    """A class for object detection in videos using YOLOv5"""
    
    def __init__(self, model_name: str = "yolov5s", confidence_threshold: float = 0.5):
        """
        Initialize the object detector with a specified model
        
        Args:
            model_name: The YOLOv5 model to use (default: 'yolov5s')
            confidence_threshold: Minimum confidence to consider a detection (default: 0.5)
        """
        self.confidence_threshold = confidence_threshold
        try:
            # Load YOLOv5 model from torch hub
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.conf = confidence_threshold  # Set confidence threshold
            
            # Use CUDA if available
            if torch.cuda.is_available():
                self.model.cuda()
            
            logger.info(f"YOLOv5 model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv5 model: {str(e)}")
            # Fallback to CPU if loading fails
            self.model = None
            raise
    
    def extract_frames(self, video_path: str, interval: int = 30) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames from a video at regular intervals
        
        Args:
            video_path: Path to the video file
            interval: Extract one frame every N frames (default: 30)
            
        Returns:
            A tuple of (frames, timestamps) where frames is a list of numpy arrays
            and timestamps is a list of timestamps in seconds
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"  - Frames: {frame_count}")
            logger.info(f"  - FPS: {fps}")
            logger.info(f"  - Duration: {duration:.2f} seconds")
            
            frames = []
            timestamps = []
            success, frame = cap.read()
            count = 0
            
            while success:
                if count % interval == 0:
                    timestamp = count / fps
                    frames.append(frame)
                    timestamps.append(timestamp)
                    logger.debug(f"Extracted frame at {timestamp:.2f} seconds")
                
                success, frame = cap.read()
                count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames, timestamps
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def detect_objects(self, frames: List[np.ndarray], target_objects: Optional[List[str]] = None) -> List[Tuple[int, str, np.ndarray]]:
        """
        Detect objects in a list of frames
        
        Args:
            frames: List of frames (numpy arrays) to process
            target_objects: Optional list of object names to filter for
            
        Returns:
            A list of tuples (frame_index, label, frame) for frames containing target objects
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        detected_objects = []
        
        for i, frame in enumerate(frames):
            try:
                # Run detection on frame
                results = self.model(frame)
                
                # Get detection results as dataframe
                detections = results.pandas().xyxy[0]
                
                # If target objects specified, filter for them
                if target_objects:
                    detections = detections[detections['name'].isin(target_objects)]
                
                # If any detections remain, add to results
                if not detections.empty:
                    # Get unique class names detected in this frame
                    labels = detections['name'].unique()
                    
                    for label in labels:
                        # Filter detections to just this label
                        label_dets = detections[detections['name'] == label]
                        
                        # Get the detection with highest confidence
                        best_det = label_dets.iloc[label_dets['confidence'].argmax()]
                        
                        # Draw bounding box on frame for visualization
                        frame_with_box = frame.copy()
                        x1, y1, x2, y2 = map(int, [best_det['xmin'], best_det['ymin'], best_det['xmax'], best_det['ymax']])
                        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_with_box, f"{label} {best_det['confidence']:.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        detected_objects.append((i, label, frame_with_box))
            
            except Exception as e:
                logger.error(f"Error detecting objects in frame {i}: {str(e)}")
        
        logger.info(f"Detected {len(detected_objects)} instances of target objects")
        return detected_objects
    
    def find_object_in_video(self, video_path: str, target_objects: List[str], frame_interval: int = 30) -> List[Tuple[float, str, np.ndarray]]:
        """
        Find specified objects in a video
        
        Args:
            video_path: Path to the video file
            target_objects: List of object names to detect
            frame_interval: Extract one frame every N frames (default: 30)
            
        Returns:
            A list of tuples (timestamp, label, frame) for frames containing target objects
        """
        # Extract frames from video
        frames, timestamps = self.extract_frames(video_path, frame_interval)
        
        # Detect objects in frames
        detected_indices = self.detect_objects(frames, target_objects)
        
        # Map detected indices to timestamps
        results = [(timestamps[i], label, frame) for i, label, frame in detected_indices]
        
        return results
    
    def detect_and_save_frames(self, 
                               video_path: str, 
                               output_dir: str, 
                               frame_interval: int = 30,
                               save_all_frames: bool = False) -> Dict[float, Dict[str, Any]]:
        """
        Detect objects in a video and save annotated frames
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output frames
            frame_interval: Extract one frame every N frames (default: 30)
            save_all_frames: Whether to save all frames or only those with detections (default: False)
            
        Returns:
            A dictionary mapping timestamps to frame metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from video
        frames, timestamps = self.extract_frames(video_path, frame_interval)
        
        # Process each frame
        frame_data = {}
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            try:
                # Run detection on frame
                results = self.model(frame)
                
                # Get detection results as dataframe
                detections = results.pandas().xyxy[0]
                
                # Format timestamp for filename
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                frame_id = f"{minutes:02d}m{seconds:02d}s"
                
                # Skip if no detections and we're not saving all frames
                if detections.empty and not save_all_frames:
                    continue
                
                # Draw bounding boxes on frame
                annotated_frame = frame.copy()
                detected_objects = []
                
                for _, det in detections.iterrows():
                    label = det['name']
                    confidence = det['confidence']
                    box = det[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, [box['xmin'], box['ymin'], box['xmax'], box['ymax']])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    detected_objects.append({
                        "label": label,
                        "confidence": float(confidence),
                        "box": [x1, y1, x2, y2]
                    })
                
                # Save the frame
                frame_path = output_path / f"frame_{frame_id}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Save the annotated frame if there are detections
                if detected_objects:
                    annotated_path = output_path / f"detected_{frame_id}.jpg"
                    cv2.imwrite(str(annotated_path), annotated_frame)
                
                # Store frame data
                frame_data[timestamp] = {
                    "frame_path": str(frame_path),
                    "annotated_path": str(annotated_path) if detected_objects else None,
                    "objects": detected_objects,
                    "object_labels": [obj["label"] for obj in detected_objects]
                }
                
                logger.debug(f"Processed frame at {frame_id}")
                
            except Exception as e:
                logger.error(f"Error processing frame at {timestamp:.2f} seconds: {str(e)}")
        
        logger.info(f"Saved {len(frame_data)} frames to {output_dir}")
        return frame_data