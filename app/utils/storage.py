"""
Storage utilities for Luna AI
Handles file operations, directory management, and storage paths
"""
import os
import shutil
import logging
from pathlib import Path
from typing import Union, List, Optional

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise
    
    return path

def ensure_directories_exist() -> None:
    """
    Create all required application directories
    """
    # Create required directories from settings
    dirs_to_create = [
        settings.UPLOAD_DIR,
        settings.FRAME_DIR,
        settings.TRANSCRIPTION_DIR,
        settings.TEMP_DIR,
    ]
    
    for dir_path in dirs_to_create:
        ensure_directory_exists(dir_path)
        logger.info(f"Created directory: {dir_path}")

def save_uploaded_file(file_content: bytes, directory: Union[str, Path], filename: str) -> Path:
    """
    Save uploaded file to the specified directory
    
    Args:
        file_content: Content of the file
        directory: Directory to save the file
        filename: Name of the file
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    dir_path = ensure_directory_exists(directory)
    
    # Create file path
    file_path = dir_path / filename
    
    # Write file content
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        raise
    
    return file_path

def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file deleted successfully, False otherwise
    """
    path = Path(file_path)
    
    try:
        if path.exists():
            path.unlink()
            logger.info(f"Deleted file: {path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {path}")
            return False
    except Exception as e:
        logger.error(f"Failed to delete file {path}: {e}")
        return False

def delete_directory(directory_path: Union[str, Path], recursive: bool = True) -> bool:
    """
    Delete a directory
    
    Args:
        directory_path: Path to the directory
        recursive: Whether to delete recursively
        
    Returns:
        True if directory deleted successfully, False otherwise
    """
    path = Path(directory_path)
    
    try:
        if path.exists():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
            logger.info(f"Deleted directory: {path}")
            return True
        else:
            logger.warning(f"Directory not found for deletion: {path}")
            return False
    except Exception as e:
        logger.error(f"Failed to delete directory {path}: {e}")
        return False

def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    Get size of a file in bytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes or None if file doesn't exist
    """
    path = Path(file_path)
    
    try:
        if path.exists():
            size = path.stat().st_size
            return size
        else:
            logger.warning(f"File not found for size check: {path}")
            return None
    except Exception as e:
        logger.error(f"Failed to get file size for {path}: {e}")
        return None

def list_files(directory_path: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in a directory with optional pattern matching
    
    Args:
        directory_path: Path to the directory
        pattern: Glob pattern for matching files
        
    Returns:
        List of Path objects for matching files
    """
    path = Path(directory_path)
    
    try:
        if path.exists():
            files = list(path.glob(pattern))
            return files
        else:
            logger.warning(f"Directory not found for listing: {path}")
            return []
    except Exception as e:
        logger.error(f"Failed to list files in {path}: {e}")
        return []