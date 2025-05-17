# app/services/transcription_cache.py
"""
Transcription caching implementation for Luna AI
"""
import os
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import time

logger = logging.getLogger("transcription.cache")

class TranscriptionCache:
    """
    Manages caching for transcriptions to avoid reprocessing the same videos
    """
    def __init__(self, cache_dir: Path):
        """Initialize the transcription cache"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Transcription cache initialized at {cache_dir}")

        # Create index file if it doesn't exist
        self.index_path = cache_dir / "index.json"
        if not self.index_path.exists():
            with open(self.index_path, "w") as f:
                json.dump({}, f)
        else:
            # Validate index file
            try:
                with open(self.index_path, "r") as f:
                    json.load(f)
            except json.JSONDecodeError:
                logger.error("Index file corrupted, creating new one")
                with open(self.index_path, "w") as f:
                    json.dump({}, f)

    def generate_cache_key(self, youtube_url: str) -> str:
        """Generate a unique cache key for a YouTube URL"""
        # Extract video ID from various YouTube URL formats
        video_id = None
        
        # Standard format: youtube.com/watch?v=VIDEO_ID
        if "v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0] if "&" in youtube_url.split("v=")[1] else youtube_url.split("v=")[1]
        # Short format: youtu.be/VIDEO_ID
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0] if "?" in youtube_url.split("youtu.be/")[1] else youtube_url.split("youtu.be/")[1]
        # Embed format: youtube.com/embed/VIDEO_ID
        elif "youtube.com/embed/" in youtube_url:
            video_id = youtube_url.split("youtube.com/embed/")[1].split("?")[0] if "?" in youtube_url.split("youtube.com/embed/")[1] else youtube_url.split("youtube.com/embed/")[1]

        if not video_id:
            # If we couldn't extract a video ID, hash the whole URL
            return hashlib.md5(youtube_url.encode()).hexdigest()

        return video_id

    def get_from_cache(self, youtube_url: str, video_id: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a transcription exists in cache and return it

        Args:
            youtube_url: YouTube URL to check
            video_id: Optional current video ID to update in cached result

        Returns:
            Tuple of (is_cached, transcription_data)
        """
        if not youtube_url:
            logger.warning("Empty YouTube URL provided to cache")
            return False, None
            
        cache_key = self.generate_cache_key(youtube_url)
        if not cache_key:
            logger.warning(f"Failed to generate cache key for {youtube_url}")
            return False, None

        # Check if we have this in our index
        try:
            if not self.index_path.exists():
                logger.warning("Index file doesn't exist")
                return False, None
                
            with open(self.index_path, "r") as f:
                try:
                    index = json.load(f)
                except json.JSONDecodeError:
                    logger.error("Index file corrupted")
                    return False, None

            if cache_key in index:
                cache_entry = index[cache_key]
                cache_file_path = self.cache_dir / f"{cache_key}.json"

                # Check if cache file exists
                if cache_file_path.exists():
                    # Check if cache is expired
                    if "expires_at" in cache_entry and isinstance(cache_entry["expires_at"], (int, float)) and cache_entry["expires_at"] < time.time():
                        logger.info(f"Cache expired for {youtube_url} (key: {cache_key})")
                        return False, None

                    # Load cached transcription
                    try:
                        with open(cache_file_path, "r") as f:
                            transcription = json.load(f)
                    except json.JSONDecodeError:
                        logger.error(f"Corrupted cache file for {cache_key}")
                        return False, None

                    # Update video_id in the cached result if provided
                    if video_id and isinstance(transcription, dict):
                        # First update the video_id field if it exists
                        if "video_id" in transcription:
                            transcription["video_id"] = video_id

                        # Also update any reference to video ID in the text if it's an error message
                        if "text" in transcription and isinstance(transcription["text"], str) and "Video ID:" in transcription["text"]:
                            # Replace the old video ID in the text with the new one
                            old_text = transcription["text"]
                            try:
                                new_text = re.sub(r'Video ID: [a-zA-Z0-9_-]+', f'Video ID: {video_id}', old_text)
                                transcription["text"] = new_text
                            except Exception as e:
                                logger.error(f"Error updating video ID in text: {str(e)}")

                        # Update any segments that might contain video ID references
                        if "segments" in transcription and isinstance(transcription["segments"], list):
                            for segment in transcription["segments"]:
                                if isinstance(segment, dict) and "text" in segment and isinstance(segment["text"], str) and "Video ID:" in segment["text"]:
                                    old_text = segment["text"]
                                    try:
                                        new_text = re.sub(r'Video ID: [a-zA-Z0-9_-]+', f'Video ID: {video_id}', old_text)
                                        segment["text"] = new_text
                                    except Exception as e:
                                        logger.error(f"Error updating video ID in segment: {str(e)}")

                    logger.info(f"Cache hit for {youtube_url} (key: {cache_key})")
                    return True, transcription
                else:
                    logger.warning(f"Cache file not found for {cache_key}")
                    # Remove from index since file doesn't exist
                    del index[cache_key]
                    with open(self.index_path, "w") as f:
                        json.dump(index, f)
            else:
                logger.info(f"No cache entry for {cache_key}")
                
        except Exception as e:
            logger.error(f"Error reading from cache: {str(e)}")

        logger.info(f"Cache miss for {youtube_url} (key: {cache_key})")
        return False, None

    def save_to_cache(self, youtube_url: str, transcription: Dict[str, Any], 
                     expires_in: int = 30 * 24 * 60 * 60) -> bool:
        """
        Save a transcription to cache

        Args:
            youtube_url: YouTube URL
            transcription: Transcription data
            expires_in: Seconds until cache expires (default: 30 days)

        Returns:
            True if saved successfully
        """
        if not youtube_url or not transcription:
            logger.warning("Empty YouTube URL or transcription provided")
            return False
            
        cache_key = self.generate_cache_key(youtube_url)
        if not cache_key:
            logger.warning(f"Failed to generate cache key for {youtube_url}")
            return False
            
        cache_file_path = self.cache_dir / f"{cache_key}.json"

        try:
            # Save transcription data
            with open(cache_file_path, "w") as f:
                json.dump(transcription, f)

            # Update index
            index = {}
            if self.index_path.exists():
                try:
                    with open(self.index_path, "r") as f:
                        index = json.load(f)
                except json.JSONDecodeError:
                    logger.error("Index file corrupted, creating new one")
                    index = {}

            expires_at = int(time.time() + expires_in)

            index[cache_key] = {
                "youtube_url": youtube_url,
                "created_at": int(time.time()),
                "expires_at": expires_at,
                "file_path": str(cache_file_path)
            }

            with open(self.index_path, "w") as f:
                json.dump(index, f)

            logger.info(f"Saved transcription to cache for {youtube_url} (key: {cache_key})")
            return True

        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            return False

    def invalidate_cache(self, youtube_url: str) -> bool:
        """
        Invalidate cache for a specific URL

        Args:
            youtube_url: YouTube URL

        Returns:
            True if cache was invalidated
        """
        if not youtube_url:
            logger.warning("Empty YouTube URL provided to invalidate")
            return False
            
        cache_key = self.generate_cache_key(youtube_url)
        if not cache_key:
            logger.warning(f"Failed to generate cache key for {youtube_url}")
            return False
            
        cache_file_path = self.cache_dir / f"{cache_key}.json"

        try:
            # Remove file if it exists
            if cache_file_path.exists():
                os.remove(cache_file_path)

            # Update index
            if not self.index_path.exists():
                return False
                
            try:
                with open(self.index_path, "r") as f:
                    index = json.load(f)
            except json.JSONDecodeError:
                logger.error("Index file corrupted")
                return False

            if cache_key in index:
                del index[cache_key]

                with open(self.index_path, "w") as f:
                    json.dump(index, f)

                logger.info(f"Invalidated cache for {youtube_url} (key: {cache_key})")
                return True
            else:
                logger.info(f"No cache entry found to invalidate for {youtube_url}")

        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")

        return False

    def clear_expired_cache(self) -> int:
        """
        Clear all expired cache entries

        Returns:
            Number of entries cleared
        """
        if not self.index_path.exists():
            logger.warning("Index file doesn't exist")
            return 0
            
        try:
            try:
                with open(self.index_path, "r") as f:
                    index = json.load(f)
            except json.JSONDecodeError:
                logger.error("Index file corrupted")
                return 0

            current_time = time.time()
            to_remove = []

            for cache_key, cache_entry in index.items():
                if "expires_at" in cache_entry and isinstance(cache_entry["expires_at"], (int, float)) and cache_entry["expires_at"] < current_time:
                    cache_file_path = self.cache_dir / f"{cache_key}.json"

                    if cache_file_path.exists():
                        try:
                            os.remove(cache_file_path)
                        except Exception as e:
                            logger.error(f"Error removing cache file {cache_file_path}: {str(e)}")

                    to_remove.append(cache_key)

            # Update index without expired entries
            for cache_key in to_remove:
                del index[cache_key]

            with open(self.index_path, "w") as f:
                json.dump(index, f)

            logger.info(f"Cleared {len(to_remove)} expired cache entries")
            return len(to_remove)

        except Exception as e:
            logger.error(f"Error clearing expired cache: {str(e)}")
            return 0


def get_transcription_path(video_id: str) -> Optional[str]:
    """
    Get the path to a transcription file for a given video ID
    
    Args:
        video_id: ID of the video to find transcription for
        
    Returns:
        Path to transcription file or None if not found
    """
    if not video_id:
        return None
        
    # Import here to avoid circular imports
    try:
        from app.config import settings
        
        # Ensure TRANSCRIPTION_DIR exists
        if not hasattr(settings, 'TRANSCRIPTION_DIR'):
            logger.error("TRANSCRIPTION_DIR not defined in settings")
            return None
            
        transcription_dir = settings.TRANSCRIPTION_DIR
        if not os.path.exists(transcription_dir):
            logger.error(f"Transcription directory {transcription_dir} does not exist")
            return None
    
        # Check standard file patterns
        possible_paths = [
            os.path.join(transcription_dir, f"transcription_{video_id}.json"),
            os.path.join(transcription_dir, f"transcription{video_id}.json"),
            os.path.join(transcription_dir, f"transcription-{video_id}.json"),
            os.path.join(transcription_dir, f"{video_id}.json"),
            os.path.join(transcription_dir, f"{video_id}_transcription.json")
        ]
        
        # If it's an upload ID, try with hash part
        if video_id and video_id.startswith("upload_"):
            parts = video_id.split("_")
            if len(parts) >= 3:
                hash_part = parts[2]
                
                # Look for files containing the hash part
                try:
                    for file in os.listdir(transcription_dir):
                        if hash_part in file and file.endswith(".json") and "transcription" in file:
                            path = os.path.join(transcription_dir, file)
                            if os.path.exists(path):
                                return path
                except Exception as e:
                    logger.error(f"Error searching for hash part in transcription files: {str(e)}")
        
        # Check each standard path
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        logger.warning(f"No transcription file found for video ID: {video_id}")
        return None
    except Exception as e:
        logger.error(f"Error in get_transcription_path: {str(e)}")
        return None