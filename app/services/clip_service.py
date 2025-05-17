"""
CLIP integration for enhanced visual understanding in Luna AI
Uses OpenAI's CLIP model for zero-shot image classification and similarity search
"""
import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import cv2
from PIL import Image

# Configure logging
logger = logging.getLogger("clip_service")

# Check if CLIP is available
try:
    import clip
    CLIP_AVAILABLE = True
    logger.info("CLIP module is available")
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP module not available - using fallbacks for visual understanding")

class CLIPService:
    """Service for using CLIP for visual understanding"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP service
        
        Args:
            model_name: Name of the CLIP model to use
        """
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        
        # Try to initialize CLIP
        if CLIP_AVAILABLE:
            try:
                # Use CUDA if available
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load(model_name, device=self.device)
                logger.info(f"CLIP model {model_name} loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Error loading CLIP model: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if CLIP service is available"""
        return self.model is not None and self.preprocess is not None
    
    def classify_image(
        self, 
        image, 
        categories: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Classify image into given categories
        
        Args:
            image: PIL Image or numpy array
            categories: List of category names
            
        Returns:
            List of (category, confidence) tuples sorted by confidence
        """
        if not self.is_available():
            logger.warning("CLIP not available, returning random classifications")
            # Return random classifications for testing
            import random
            results = [(cat, random.random()) for cat in categories]
            return sorted(results, key=lambda x: x[1], reverse=True)
        
        try:
            # Convert numpy image to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Ensure image is in the right format for CLIP
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Format categories with a prompt template
            texts = [f"a photo of {category}" for category in categories]
            
            # Encode text prompts
            text = clip.tokenize(texts).to(self.device)
            
            # Calculate similarity
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Match categories with probabilities
            results = list(zip(categories, probs))
            
            # Sort by probability
            return sorted(results, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error classifying image with CLIP: {str(e)}")
            # Return equal probabilities as fallback
            return [(cat, 1.0 / len(categories)) for cat in categories]
    
    def describe_image(self, image) -> Dict[str, Any]:
        """
        Generate detailed descriptions for an image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with scene type, attributes, and objects
        """
        if not self.is_available():
            logger.warning("CLIP not available, returning generic description")
            return {
                "scene_type": "unknown",
                "attributes": [],
                "objects": []
            }
        
        try:
            # Scene type classification
            scene_types = [
                "indoor", "outdoor", "natural landscape", "urban", "restaurant", 
                "office", "home", "street", "beach", "forest", "mountain", 
                "desert", "snow", "water", "night scene", "daytime scene"
            ]
            scene_results = self.classify_image(image, scene_types)
            
            # Attribute classification
            attributes = [
                "bright", "dark", "colorful", "monochrome", "blurry", "clear",
                "crowded", "empty", "modern", "vintage", "stylish", "plain",
                "formal", "casual", "natural", "artificial", "warm", "cool"
            ]
            attribute_results = self.classify_image(image, attributes)
            
            # Object classification
            common_objects = [
                "person", "chair", "table", "car", "building", "tree", "plant",
                "food", "drink", "computer", "phone", "dog", "cat", "book",
                "painting", "sculpture", "window", "door", "light", "shadow"
            ]
            object_results = self.classify_image(image, common_objects)
            
            # Filter by confidence threshold
            threshold = 0.2
            
            top_scene = scene_results[0][0] if scene_results[0][1] > threshold else "unknown"
            top_attributes = [attr for attr, conf in attribute_results if conf > threshold][:3]
            top_objects = [obj for obj, conf in object_results if conf > threshold][:5]
            
            return {
                "scene_type": top_scene,
                "attributes": top_attributes,
                "objects": top_objects
            }
            
        except Exception as e:
            logger.error(f"Error describing image with CLIP: {str(e)}")
            return {
                "scene_type": "unknown",
                "attributes": [],
                "objects": []
            }
    
    def create_image_embedding(self, image) -> Optional[np.ndarray]:
        """
        Create an embedding vector for an image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.is_available():
            logger.warning("CLIP not available, returning None for image embedding")
            return None
        
        try:
            # Convert numpy image to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Ensure image is in the right format for CLIP
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                
            # Convert to numpy and normalize
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating image embedding: {str(e)}")
            return None
    
    def create_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Create an embedding vector for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.is_available():
            logger.warning("CLIP not available, returning None for text embedding")
            return None
        
        try:
            # Tokenize the text
            text_token = clip.tokenize([text]).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.model.encode_text(text_token)
                
            # Convert to numpy and normalize
            embedding = text_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating text embedding: {str(e)}")
            return None
    
    def compare_image_to_text(self, image, text: str) -> float:
        """
        Calculate similarity between image and text
        
        Args:
            image: PIL Image or numpy array
            text: Text to compare against
            
        Returns:
            Similarity score (0-1)
        """
        if not self.is_available():
            logger.warning("CLIP not available, returning random similarity")
            import random
            return random.random()
        
        try:
            # Convert numpy image to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess image
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_token = clip.tokenize([text]).to(self.device)
            
            # Calculate similarity
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text_token)
                similarity = logits_per_image.softmax(dim=-1).cpu().numpy()[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing image to text: {str(e)}")
            return 0.5  # Neutral fallback
    
    def find_best_matching_frame(
        self, 
        frames: List[Dict[str, Any]], 
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find the frame that best matches a text query
        
        Args:
            frames: List of frame dictionaries with 'path' key
            query: Text query to match against
            
        Returns:
            Best matching frame or None if no matches
        """
        if not self.is_available() or not frames:
            logger.warning("Cannot find matching frame: CLIP not available or no frames provided")
            return frames[0] if frames else None
        
        try:
            best_match = None
            best_score = -1
            
            # Calculate similarity for each frame
            for frame in frames:
                # Skip frames without path
                path = frame.get("path")
                if not path or not os.path.exists(path):
                    continue
                
                # Load image
                image = Image.open(path).convert("RGB")
                
                # Compare to query
                score = self.compare_image_to_text(image, query)
                
                # Update best match
                if score > best_score:
                    best_score = score
                    best_match = frame
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding best matching frame: {str(e)}")
            return frames[0] if frames else None


# Singleton instance
_clip_service = None

def get_clip_service() -> CLIPService:
    """
    Get or create the singleton instance of CLIPService
    
    Returns:
        CLIPService instance
    """
    global _clip_service
    if _clip_service is None:
        _clip_service = CLIPService()
    return _clip_service