"""
Timestamp utilities for Luna AI
Provides functions for working with video timestamps and time-based data
"""
import re
from typing import Dict, Any, List, Tuple, Optional, Union


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into MM:SS or HH:MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if isinstance(seconds, str):
        try:
            seconds = float(seconds)
        except ValueError:
            return "00:00"
            
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse a timestamp string to seconds
    
    Args:
        timestamp_str: Timestamp string (e.g., "1:30", "01:30", "1:30:45")
        
    Returns:
        Time in seconds or 0 if invalid
    """
    if not timestamp_str:
        return 0
        
    try:
        # Split by colons
        parts = timestamp_str.strip().split(':')
        
        # Handle different formats
        if len(parts) == 3:  # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 1:  # SS
            return int(parts[0])
        else:
            return 0
    except (ValueError, IndexError):
        return 0


def extract_timestamp_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract timestamp information from text
    
    Args:
        text: Text that might contain timestamp references
        
    Returns:
        Dictionary with timestamp information or None if not found
    """
    # Check for HH:MM:SS format
    hhmmss_pattern = r'(\d+):(\d+):(\d+)'
    match = re.search(hhmmss_pattern, text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        time_seconds = hours * 3600 + minutes * 60 + seconds
        return {
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "total_seconds": time_seconds,
            "formatted": f"{hours}:{minutes:02d}:{seconds:02d}"
        }
    
    # Check for MM:SS format
    mmss_pattern = r'(\d+):(\d+)'
    match = re.search(mmss_pattern, text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        time_seconds = minutes * 60 + seconds
        return {
            "hours": 0,
            "minutes": minutes,
            "seconds": seconds,
            "total_seconds": time_seconds,
            "formatted": f"{minutes}:{seconds:02d}"
        }
    
    # Check for "X minutes Y seconds" format
    time_pattern = r'(?:(\d+)\s*hours?)?[,\s]*(?:(\d+)\s*minutes?)?[,\s]*(?:(\d+)\s*seconds?)?'
    match = re.search(time_pattern, text)
    if match and any(match.groups()):
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        
        if hours > 0 or minutes > 0 or seconds > 0:
            time_seconds = hours * 3600 + minutes * 60 + seconds
            
            if hours > 0:
                formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                formatted = f"{minutes}:{seconds:02d}"
                
            return {
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
                "total_seconds": time_seconds,
                "formatted": formatted
            }
    
    return None


def is_timestamp_question(question: str) -> bool:
    """
    Check if a question is asking about a specific timestamp
    
    Args:
        question: User's question
        
    Returns:
        True if it's a timestamp question, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    question = question.lower()
    
    # Check for direct timestamp patterns
    timestamp_patterns = [
        r'\b(?:at|around|near|approximately|about) \d+:\d+',
        r'\b(?:at|around|near|approximately|about) \d+ (?:second|minute)s?',
        r'\b(?:at|around|near|approximately|about) the \d+ (?:second|minute)',
        r'\btime(?:stamp)? \d+:\d+',
        r'\bsecond \d+',
        r'\bminute \d+',
        r'\d+:\d+'
    ]
    
    for pattern in timestamp_patterns:
        if re.search(pattern, question):
            return True
    
    # Check for "when" questions about visual events
    when_patterns = [
        r'\bwhen\b.*\bshown\b',
        r'\bwhen\b.*\bappears?\b',
        r'\bwhen\b.*\bhappens?\b',
        r'\bwhen\b.*\bsee\b',
        r'\bwhen\b.*\bdisplayed\b',
        r'\bwhen\b.*\bvisible\b',
        r'\bat what (?:time|point|moment)\b'
    ]
    
    for pattern in when_patterns:
        if re.search(pattern, question):
            return True
    
    return False


def find_nearest_timestamp(
    target_time: float, 
    timestamps: List[Union[float, Dict[str, Any]]]
) -> Optional[Union[float, Dict[str, Any]]]:
    """
    Find the nearest timestamp to a target time
    
    Args:
        target_time: Target time in seconds
        timestamps: List of timestamps (either float seconds or dicts with 'time' key)
        
    Returns:
        Nearest timestamp or None if empty list
    """
    if not timestamps:
        return None
    
    # Handle different timestamp formats
    def get_time(ts):
        if isinstance(ts, dict):
            return ts.get('time', 0)
        return ts
    
    # Find nearest
    nearest = min(timestamps, key=lambda x: abs(get_time(x) - target_time))
    return nearest


def create_timestamp_index(timestamps: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Create an index of timestamps by second for faster lookup
    
    Args:
        timestamps: List of timestamp dictionaries with 'time' key
        
    Returns:
        Dictionary mapping seconds to lists of timestamp dictionaries
    """
    index = {}
    
    for ts in timestamps:
        if 'time' not in ts:
            continue
        
        # Convert to int for indexing
        second = int(ts['time'])
        
        if second not in index:
            index[second] = []
        
        index[second].append(ts)
    
    return index


def get_timestamp_range_str(start_time: float, end_time: float) -> str:
    """
    Format a timestamp range as a string
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Formatted timestamp range string
    """
    start_str = format_timestamp(start_time)
    end_str = format_timestamp(end_time)
    return f"{start_str} - {end_str}"


def extract_all_timestamps(text: str) -> List[Dict[str, Any]]:
    """
    Extract all timestamps mentioned in a text
    
    Args:
        text: Input text
        
    Returns:
        List of timestamp dictionaries
    """
    timestamps = []
    
    # Search for HH:MM:SS format
    hhmmss_matches = re.finditer(r'(\d+):(\d+):(\d+)', text)
    for match in hhmmss_matches:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        time_seconds = hours * 3600 + minutes * 60 + seconds
        
        timestamps.append({
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "total_seconds": time_seconds,
            "formatted": f"{hours}:{minutes:02d}:{seconds:02d}",
            "position": match.start()
        })
    
    # Search for MM:SS format
    mmss_matches = re.finditer(r'(\d+):(\d+)', text)
    for match in mmss_matches:
        # Skip if this is part of an HH:MM:SS format (already captured)
        if match.start() > 0 and text[match.start()-1] == ':':
            continue
        
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        time_seconds = minutes * 60 + seconds
        
        timestamps.append({
            "hours": 0,
            "minutes": minutes,
            "seconds": seconds,
            "total_seconds": time_seconds,
            "formatted": f"{minutes}:{seconds:02d}",
            "position": match.start()
        })
    
    # Sort by position in text
    return sorted(timestamps, key=lambda x: x["position"])