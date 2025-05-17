# app/services/ai.py
"""
Enhanced AI service for Luna AI
Handles AI-powered video analysis and chat responses using Google's Gemini
with timestamp support, contextual Q&A, and semantic search
"""
import os
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from pathlib import Path

# Configure logging
logger = logging.getLogger("ai_service")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Create a file handler if log directory exists
log_dir = os.environ.get("LOG_DIR", "/tmp")
if os.path.exists(log_dir):
    file_handler = logging.FileHandler(os.path.join(log_dir, "ai_service.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Import settings
from app.config import settings

# Initialize Gemini API if available
try:
    import google.generativeai as genai
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if API_KEY:
        genai.configure(api_key=API_KEY)
        # Default to Gemini 1.5 Pro for best performance with long text
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        # Initialize vision model for image analysis if needed
        vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        logger.info("Gemini API initialized successfully")
    else:
        logger.warning("GEMINI_API_KEY not found in environment variables")
        gemini_model = None
        vision_model = None
except ImportError:
    logger.warning("google.generativeai package not found")
    gemini_model = None
    vision_model = None
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    gemini_model = None
    vision_model = None

# Cache for AI responses to improve performance
AI_CACHE = {}

async def generate_ai_response(
    video_id: str,
    question: str,
    tab_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate AI response to a question about a video using Gemini
    with timestamp support
    
    Args:
        video_id: ID of the video to analyze
        question: User's question
        tab_id: Optional tab ID for session tracking
        
    Returns:
        Dictionary with AI response and optional timestamps
    """
    start_time = time.time()
    cache_key = f"{video_id}:{question}"
    
    # Check cache first
    if cache_key in AI_CACHE:
        logger.info(f"Cache hit for question: '{question}'")
        return AI_CACHE[cache_key]
    
    logger.info(f"Generating AI response for video {video_id}, question: '{question}'")
    
    try:
        # Load transcription
        transcription = await load_transcription(video_id)
        if not transcription or len(transcription) < 10:
            logger.warning(f"No valid transcription found for video_id: {video_id}")
            return {
                "answer": f"I couldn't find a transcript for this video (ID: {video_id}). Please ensure the video has been processed first.",
                "timestamps": []
            }
        
        # Load timestamps
        timestamps_data = await get_timestamps_for_video(video_id)
        has_timestamps = bool(timestamps_data and timestamps_data.get("formatted_timestamps"))
        
        # Try to get visual service if available
        try:
            from app.services.visual_analysis import get_visual_analysis_service
            visual_service = get_visual_analysis_service()
            visual_data = await visual_service.load_visual_data(video_id) if visual_service else {}
        except (ImportError, AttributeError):
            visual_service = None
            visual_data = {}
            
        # Analyze question to determine best approach
        question_lower = question.lower()
        
        # Check if it's a timing question
        is_timing_question = any(keyword in question_lower for keyword in [
            'when', 'time', 'timestamp', 'moment', 'at what point', 'what time', 
            'which part', 'during', 'start', 'end', 'beginning'
        ])
        
        # Process with timestamp awareness
        answer, response_timestamps = await process_question_with_timestamps(
            question=question,
            transcription=transcription,
            timestamps_data=timestamps_data,
            visual_data=visual_data,
            is_timing_question=is_timing_question
        )
        
        # Create the result
        result = {
            "answer": answer,
            "timestamps": response_timestamps,
            "question": question,
            "video_id": video_id
        }
        
        # Cache result for future use
        AI_CACHE[cache_key] = result
        
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"Generated response in {elapsed:.2f} seconds")
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        return {
            "answer": f"I encountered an error while processing your question. Please try again or ask a different question.",
            "timestamps": [],
            "error": str(e)
        }

async def generate_ai_response_with_history(
    video_id: str,
    question: str,
    conversation_history: List[Dict] = None,
    tab_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate AI response to a question about a video with conversation history
    
    Args:
        video_id: ID of the video
        question: User's question
        conversation_history: Previous conversation turns
        tab_id: Optional tab ID
        
    Returns:
        Dictionary with answer and relevant timestamps
    """
    start_time = time.time()
    logger.info(f"Generating AI response with conversation history for question: '{question}', video_id: {video_id}")
    
    if conversation_history is None:
        conversation_history = []
    
    try:
        # Get transcript for the video
        transcription = await load_transcription(video_id)
        if not transcription or transcription.startswith("No transcription found") or len(transcription) < 10:
            logger.warning(f"No valid transcript found for video {video_id}")
            return {
                "answer": f"I couldn't find a transcript for this video. Please ensure the video has been processed first.",
                "timestamps": []
            }
        
        # Get timestamps for the video if available
        timestamps_data = await get_timestamps_for_video(video_id)
        has_timestamps = bool(timestamps_data and timestamps_data.get("formatted_timestamps"))
        
        # Create cache key including conversation context
        history_hash = hash(str(conversation_history[-3:]) if conversation_history else "")
        cache_key = f"{video_id}_{hash(question)}_{history_hash}"
        
        # Check cache for quick response
        if cache_key in AI_CACHE:
            cached_result = AI_CACHE[cache_key]
            logger.info(f"Using cached AI response for '{question}'")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generated cached response in {elapsed_time:.2f} seconds")
            return cached_result
        
        # Try to get visual service if available
        try:
            from app.services.visual_analysis import get_visual_analysis_service
            visual_service = get_visual_analysis_service()
            visual_data = await visual_service.load_visual_data(video_id) if visual_service else {}
        except (ImportError, AttributeError):
            visual_service = None
            visual_data = {}
        
        # Prepare conversation context
        conversation_context = ""
        if conversation_history:
            # Format previous conversation turns
            for turn in conversation_history[-5:]:  # Use last 5 turns to keep context manageable
                if 'user' in turn:
                    conversation_context += f"User: {turn['user']}\n"
                if 'assistant' in turn:
                    conversation_context += f"Assistant: {turn['assistant']}\n"
            
            conversation_context += "\n"
        
        # Check if it's a timing question
        question_lower = question.lower()
        is_timing_question = any(keyword in question_lower for keyword in [
            'when', 'time', 'timestamp', 'moment', 'at what point', 'what time', 
            'which part', 'during', 'start', 'end', 'beginning'
        ])
        
        # Process with timestamp awareness and conversation history
        answer, response_timestamps = await process_question_with_timestamps(
            question=question,
            transcription=transcription,
            timestamps_data=timestamps_data,
            visual_data=visual_data,
            is_timing_question=is_timing_question,
            conversation_history=conversation_history
        )
        
        # Create the final response
        result = {
            "answer": answer,
            "timestamps": response_timestamps,
            "question": question,
            "video_id": video_id
        }
        
        # Store in cache for future queries
        AI_CACHE[cache_key] = result
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Generated AI response with conversation history in {elapsed_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating AI response with history: {str(e)}", exc_info=True)
        return {
            "answer": f"I encountered an error while processing your question. Please try again or ask a different question.",
            "timestamps": [],
            "error": str(e)
        }

async def process_question_with_timestamps(
    question: str,
    transcription: str,
    timestamps_data: Dict[str, Any],
    visual_data: Dict[str, Any] = None,
    is_timing_question: bool = False,
    conversation_history: List[Dict] = None
) -> Tuple[str, List[Dict]]:
    """
    Process a question with timestamp awareness
    
    Args:
        question: User's question
        transcription: Video transcript
        timestamps_data: Timestamp data
        visual_data: Optional visual analysis data
        is_timing_question: Whether the question is about timing
        conversation_history: Optional conversation history
        
    Returns:
        Tuple of (answer text, timestamps list)
    """
    # Extract keywords from the question
    keywords = extract_keywords(question)
    logger.info(f"Extracted keywords: {keywords}")
    
    # Find relevant sections based on keywords
    formatted_timestamps = timestamps_data.get("formatted_timestamps", [])
    relevant_sections, timestamp_ranges = extract_relevant_sections(
        transcription=transcription, 
        keywords=keywords,
        formatted_timestamps=formatted_timestamps
    )
    
    # Prepare the answer
    answer = ""
    response_timestamps = []
    
    # For Gemini API
    if gemini_model:
        # Create a prompt including timestamps
        timestamp_instruction = ""
        if formatted_timestamps:
            timestamp_instruction = """
            Timestamps are available for this video. When relevant, include reference to timestamps 
            using the MM:SS format to help the user navigate to specific parts of the video.
            """
            
        # Prepare conversation context
        context_section = ""
        if conversation_history and len(conversation_history) > 0:
            # Format previous conversation turns
            context_section += "Previous conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                if 'user' in turn:
                    context_section += f"User: {turn['user']}\n"
                if 'assistant' in turn:
                    context_section += f"Assistant: {turn['assistant']}\n"
            context_section += "\n"
        
        # Prepare relevant transcript sections
        transcript_section = ""
        if relevant_sections:
            transcript_section = "Relevant transcript sections:\n" + "\n".join(relevant_sections)
        else:
            # If no relevant sections found, use a portion of the transcript
            transcript_section = "Transcript excerpt:\n" + transcription[:5000]
        
        # Create a prompt for Gemini
        prompt = f"""You are an AI assistant answering questions about a video based on its transcript. 
        {timestamp_instruction}
        
        {context_section}
        {transcript_section}
        
        Question: {question}
        
        Answer the question based on the transcript information. Be concise but thorough.
        If the answer is not in the transcript, say so clearly rather than making up information.
        For timing questions, always reference specific timestamps (MM:SS format) when available.
        """
        
        try:
            # Call Gemini API
            response = gemini_model.generate_content(
                prompt, 
                generation_config={"temperature": 0.2, "max_output_tokens": 800}
            )
            answer = response.text
            
            # Extract timestamps mentioned in the response
            response_timestamps = extract_mentioned_timestamps(answer)
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            answer = f"I encountered an error processing your question about the video. Please try again with a different question."
    else:
        # Fallback if Gemini is not available
        if relevant_sections:
            answer = f"Based on the video transcript, I found these relevant sections that may answer your question:\n\n"
            for i, section in enumerate(relevant_sections[:3]):
                answer += f"{i+1}. {section}\n\n"
        else:
            answer = f"I found this information in the video transcript that might be relevant to your question:\n\n"
            # Just use the first 200 chars of the transcript as a simple fallback
            answer += transcription[:200] + "..."
    
    # If no timestamps were directly mentioned but we have timestamp data
    if not response_timestamps and formatted_timestamps and (is_timing_question or timestamp_ranges):
        # For timing questions, add timestamps from our analysis
        if timestamp_ranges:
            for start_time, end_time, text_snippet in timestamp_ranges[:3]:  # Top 3 most relevant
                time_obj = {
                    "time": start_time,
                    "time_formatted": format_seconds_to_mmss(start_time),
                    "text": text_snippet[:100] + "..."  # First 100 chars
                }
                if time_obj not in response_timestamps:
                    response_timestamps.append(time_obj)
            
            # Add note about found timestamps if none were mentioned in the answer
            if "timestamp" not in answer.lower() and response_timestamps:
                note = "\n\nI found relevant information at these timestamps: "
                for ts in response_timestamps:
                    note += f"\n• {ts['time_formatted']}: {ts['text']}"
                
                answer += note
    
    # Visual data integration
    if visual_data and not response_timestamps:
        try:
            # Try to extract timestamps from scenes
            scenes = visual_data.get("scenes", [])
            if scenes:
                scene_timestamps = []
                for scene in scenes[:2]:  # Top 2 scenes
                    if "start_time" in scene and "description" in scene:
                        scene_time = scene.get("start_time", 0)
                        scene_timestamps.append({
                            "time": scene_time,
                            "time_formatted": format_seconds_to_mmss(scene_time),
                            "text": scene.get("description", "Notable scene")
                        })
                
                # Only add these if we don't have other timestamps
                if not response_timestamps:
                    response_timestamps = scene_timestamps
        except Exception as visual_err:
            logger.error(f"Error processing visual data: {str(visual_err)}")
    
    return answer, response_timestamps

async def summarize_video(video_id: str) -> str:
    """
    Generate a summary of the video content
    
    Args:
        video_id: ID of the video
        
    Returns:
        Summary text
    """
    try:
        # Load transcript
        transcript = await load_transcription(video_id)
        if not transcript or transcript.startswith("No transcription found") or len(transcript) < 10:
            return "I couldn't find a transcript for this video. Please ensure the video has been processed first."
        
        # Limit transcript length for model context
        if len(transcript) > 10000:
            transcript = transcript[:10000] + "..."
        
        # Use Gemini if available
        if gemini_model:
            # Create prompt for summarization
            prompt = f"""
            Please create a concise summary of the following video transcript. Focus on the main points,
            key ideas, and conclusions. Keep the summary clear and informative.
            
            Transcript:
            {transcript}
            """
            
            # Call language model
            response = gemini_model.generate_content(
                prompt, 
                generation_config={"temperature": 0.2, "max_output_tokens": 800}
            )
            
            return response.text
        else:
            # Simple fallback summary
            return f"The video transcript contains information about: {transcript[:200]}..."
        
    except Exception as e:
        logger.error(f"Error generating video summary: {str(e)}", exc_info=True)
        return "I encountered an error while trying to summarize this video. Please try again later."

async def analyze_topics(transcript: str, visual_data: Dict = None) -> str:
    """
    Analyze topics in the video content
    
    Args:
        transcript: Video transcript
        visual_data: Optional visual analysis data
        
    Returns:
        Topics analysis text
    """
    try:
        # Limit transcript length for model context
        if len(transcript) > 10000:
            transcript = transcript[:10000] + "..."
        
        # Use Gemini if available
        if gemini_model:
            # Create prompt for topic analysis
            prompt = f"""
            Please analyze the following video transcript and identify the main topics discussed.
            For each topic, provide:
            - A brief title
            - A short description of what was discussed
            
            Format each topic as:
            - [Topic Title]: [Description]
            
            Transcript:
            {transcript}
            """
            
            # Add visual data if available
            if visual_data:
                prompt += f"\n\nAdditional visual information about the video: {json.dumps(visual_data)[:1000]}"
            
            # Call language model
            response = gemini_model.generate_content(
                prompt, 
                generation_config={"temperature": 0.2, "max_output_tokens": 600}
            )
            
            return response.text
        else:
            # Simple fallback
            return "- Main topic: Video content\n- Secondary topics: Various subjects discussed"
        
    except Exception as e:
        logger.error(f"Error analyzing topics: {str(e)}", exc_info=True)
        return "- Various topics: The video covers multiple subjects\n- Unable to analyze in detail due to a technical error"

async def load_transcription(video_id: str) -> str:
    """
    Load transcription from saved file with improved error handling
    
    Args:
        video_id: ID of the video
        
    Returns:
        Transcript text or empty string if not found
    """
    transcription_paths = [
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json")
    ]
    
    # Try each possible file path
    for path in transcription_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Safely get text with default
                        return data.get("text", "")
        except Exception as e:
            logger.error(f"Error loading transcription from {path}: {str(e)}")
    
    # If this is an upload ID, try to find by hash part
    if "upload_" in video_id and "_" in video_id:
        hash_part = video_id.split("_")[-1]
        if hash_part:
            try:
                # Search for files with this hash part
                for file in os.listdir(settings.TRANSCRIPTION_DIR):
                    if hash_part in file and file.endswith(".json"):
                        file_path = os.path.join(settings.TRANSCRIPTION_DIR, file)
                        try:
                            with open(file_path, "r") as f:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    return data.get("text", "")
                        except Exception as e:
                            logger.error(f"Error reading potential match file: {str(e)}")
            except Exception as e:
                logger.error(f"Error searching for files by hash: {str(e)}")
    
    # Try database as fallback
    try:
        from sqlalchemy import text
        from app.utils.database import get_db_context
        with get_db_context() as db:
            result = db.execute(
                text("SELECT transcription FROM videos WHERE id::text = :id"),
                {"id": str(video_id)}
            )
            video_row = result.fetchone()
            if video_row and video_row[0]:
                return video_row[0]
            
            # If not found and this is an upload ID, try searching by hash part
            if "upload_" in video_id and "_" in video_id:
                hash_part = video_id.split("_")[-1]
                if hash_part:
                    like_result = db.execute(
                        text("SELECT transcription FROM videos WHERE id::text LIKE :pattern"),
                        {"pattern": f"%{hash_part}%"}
                    )
                    like_row = like_result.fetchone()
                    
                    if like_row and like_row[0]:
                        return like_row[0]
    except Exception as e:
        logger.error(f"Error loading transcription from database: {str(e)}")
    
    return ""

async def get_timestamps_for_video(video_id: str) -> Dict:
    """
    Get timestamps for a video from file
    
    Args:
        video_id: ID of the video
        
    Returns:
        Dictionary with timestamp data or empty dict if not found
    """
    # Try finding timestamps file
    timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
    
    if os.path.exists(timestamp_path):
        try:
            with open(timestamp_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading timestamps file: {str(e)}")
    
    # If not found by direct ID, try finding by hash part for upload IDs
    if "upload_" in video_id and "_" in video_id:
        hash_part = video_id.split("_")[-1]
        if hash_part:
            try:
                timestamp_dir = Path(os.path.join(settings.TRANSCRIPTION_DIR, "timestamps"))
                potential_files = list(timestamp_dir.glob(f"*{hash_part}*_timestamps.json"))
                
                for pot_file in potential_files:
                    try:
                        with open(pot_file, "r") as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading potential timestamp file: {str(e)}")
            except Exception as e:
                logger.error(f"Error searching for timestamp files by hash: {str(e)}")
    
    # If timestamps aren't found, try to extract them from transcription file
    transcription_paths = [
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription_{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription-{video_id}.json"),
        os.path.join(settings.TRANSCRIPTION_DIR, f"transcription{video_id}.json")
    ]
    
    for path in transcription_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if "timestamps" in data and data["timestamps"]:
                        # Format timestamps for display
                        formatted_timestamps = format_timestamps_for_display(data["timestamps"])
                        
                        # Save to a separate file for future use
                        os.makedirs(os.path.join(settings.TRANSCRIPTION_DIR, "timestamps"), exist_ok=True)
                        timestamp_path = os.path.join(settings.TRANSCRIPTION_DIR, "timestamps", f"{video_id}_timestamps.json")
                        
                        result = {
                            "raw_timestamps": data["timestamps"],
                            "formatted_timestamps": formatted_timestamps
                        }
                        
                        with open(timestamp_path, "w") as f:
                            json.dump(result, f)
                            
                        return result
            except Exception as e:
                logger.error(f"Error extracting timestamps from transcription: {str(e)}")
    
    # If we get here, no timestamps were found
    return {}

def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from the query text
    
    Args:
        text: Input query text
        
    Returns:
        List of extracted keywords
    """
    # Remove common stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'can',
        'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must',
        'to', 'of', 'in', 'for', 'on', 'by', 'at', 'from', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now'
    }
    
    # Normalize text
    text = text.lower()
    
    # Extract words
    words = re.findall(r'\b\w+\b', text)
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Add multi-word phrases that might be important (bigrams)
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stop_words or words[i+1] not in stop_words:
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 5:  # Only add substantial bigrams
                bigrams.append(bigram)
    
    # Combine keywords and bigrams
    all_keywords = keywords + bigrams
    
    # Prioritize keywords in quotes if they exist
    quoted_phrases = re.findall(r'"([^"]+)"', text)
    all_keywords.extend(quoted_phrases)
    
    # Deduplicate
    unique_keywords = list(set(all_keywords))
    
    # Sort by length (longer words are often more specific/important)
    unique_keywords.sort(key=len, reverse=True)
    
    return unique_keywords[:10]  # Return top 10 keywords

def extract_relevant_sections(
    transcription: str, 
    keywords: List[str],
    formatted_timestamps: List[Dict] = None
) -> Tuple[List[str], List[Tuple]]:
    """
    Extract sections of the transcript most relevant to the question
    
    Args:
        transcription: Full transcript text
        keywords: Keywords from the question
        formatted_timestamps: Optional list of formatted timestamps
        
    Returns:
        Tuple of (list of relevant text sections, list of relevant timestamp ranges)
    """
    # If no keywords provided, split by paragraphs or sentences
    if not keywords:
        sections = re.split(r'(?<=[.!?])\s+', transcription)
        return sections[:5], []  # Just return first 5 sections as a fallback
    
    # If we have timestamps, use them for more precise extraction
    if formatted_timestamps:
        return extract_sections_with_timestamps(transcription, keywords, formatted_timestamps)
    
    # Simple approach: split transcript into paragraphs or sentences
    sections = re.split(r'(?<=[.!?])\s+', transcription)
    
    # Score each section based on keyword matches
    scored_sections = []
    for section in sections:
        score = 0
        for keyword in keywords:
            if keyword.lower() in section.lower():
                score += 1
        
        if score > 0:
            scored_sections.append((section, score))
    
    # Sort by score and get top sections
    scored_sections.sort(key=lambda x: x[1], reverse=True)
    top_sections = [section for section, score in scored_sections[:5]]
    
    # If we don't have enough relevant sections, include some context
    if len(top_sections) < 3 and len(sections) > 5:
        # Add some surrounding context
        for section, score in scored_sections[:2]:
            if section in top_sections:
                idx = sections.index(section)
                if idx > 0 and sections[idx-1] not in top_sections:
                    top_sections.append(sections[idx-1])
                if idx < len(sections) - 1 and sections[idx+1] not in top_sections:
                    top_sections.append(sections[idx+1])
    
    return top_sections, []

def extract_sections_with_timestamps(
    transcription: str, 
    keywords: List[str],
    formatted_timestamps: List[Dict]
) -> Tuple[List[str], List[Tuple]]:
    """
    Extract relevant sections using timestamp information
    
    Args:
        transcription: Full transcript text
        keywords: List of keywords from the question
        formatted_timestamps: List of formatted timestamps
        
    Returns:
        Tuple of (list of relevant text sections, list of relevant timestamp tuples)
    """
    relevant_sections = []
    timestamp_ranges = []
    
    # Score each timestamp section based on keyword matches
    for ts_section in formatted_timestamps:
        section_text = ts_section.get("text", "")
        start_time = ts_section.get("start_time", 0)
        end_time = ts_section.get("end_time", 0)
        
        score = 0
        for keyword in keywords:
            if keyword.lower() in section_text.lower():
                score += 1
        
        if score > 0:
            relevant_sections.append(section_text)
            timestamp_ranges.append((start_time, end_time, section_text))
    
    # Sort timestamp ranges by score (length of tuple is a proxy for score)
    timestamp_ranges.sort(key=lambda x: sum(1 for keyword in keywords if keyword.lower() in x[2].lower()), reverse=True)
    
    # If we haven't found enough relevant sections, include some broader context
    if len(relevant_sections) < 3:
        # Group consecutive timestamp sections to get more context
        grouped_sections = []
        current_group = []
        
        for i, ts_section in enumerate(formatted_timestamps):
            has_match = any(keyword.lower() in ts_section.get("text", "").lower() for keyword in keywords)
            
            if has_match:
                # Add previous section for context if possible
                if i > 0 and formatted_timestamps[i-1] not in current_group:
                    current_group.append(formatted_timestamps[i-1])
                
                # Add the matching section
                current_group.append(ts_section)
                
                # Add next section for context if possible
                if i < len(formatted_timestamps) - 1 and formatted_timestamps[i+1] not in current_group:
                    current_group.append(formatted_timestamps[i+1])
            elif current_group:
                # Finish the current group
                grouped_sections.append(current_group)
                current_group = []
        
        # Add the last group if not empty
        if current_group:
            grouped_sections.append(current_group)
        
        # Extract text from grouped sections
        for group in grouped_sections:
            group_text = " ".join(section.get("text", "") for section in group)
            if group_text not in relevant_sections:
                relevant_sections.append(group_text)
                
                # Get timestamp range for the group
                start_time = group[0].get("start_time", 0)
                end_time = group[-1].get("end_time", 0)
                timestamp_ranges.append((start_time, end_time, group_text))
    
    # If still not enough relevant sections, add some general sections
    if len(relevant_sections) < 2:
        # Add introduction (first few sections)
        intro_text = " ".join(section.get("text", "") for section in formatted_timestamps[:3])
        if intro_text and intro_text not in relevant_sections:
            relevant_sections.append(intro_text)
            timestamp_ranges.append((
                formatted_timestamps[0].get("start_time", 0),
                formatted_timestamps[2].get("end_time", 0),
                intro_text
            ))
        
        # Add conclusion (last few sections)
        if len(formatted_timestamps) > 3:
            conclusion_text = " ".join(section.get("text", "") for section in formatted_timestamps[-3:])
            if conclusion_text and conclusion_text not in relevant_sections:
                relevant_sections.append(conclusion_text)
                timestamp_ranges.append((
                    formatted_timestamps[-3].get("start_time", 0),
                    formatted_timestamps[-1].get("end_time", 0),
                    conclusion_text
                ))
    
    return relevant_sections, timestamp_ranges

def extract_mentioned_timestamps(text: str) -> List[Dict]:
    """
    Extract timestamps mentioned in the AI response
    
    Args:
        text: AI response text
        
    Returns:
        List of timestamp dictionaries
    """
    # Look for timestamp patterns in the format MM:SS or HH:MM:SS
    timestamp_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
    
    timestamps = []
    matches = re.finditer(timestamp_pattern, text)
    
    for match in matches:
        # Get the timestamp parts
        minutes, seconds = match.group(1), match.group(2)
        hours = match.group(3) if match.group(3) else '0'
        
        # Convert to seconds
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        
        # Get some surrounding context
        start_pos = max(0, match.start() - 40)
        end_pos = min(len(text), match.end() + 80)
        context = text[start_pos:end_pos].strip()
        
        # Format time for display
        formatted_time = f"{minutes}:{seconds}"
        
        timestamps.append({
            'time': total_seconds,
            'time_formatted': formatted_time,
            'text': context
        })
    
    return timestamps

def format_timestamps_for_display(timestamps: List[Dict]) -> List[Dict]:
    """
    Format raw timestamps into user-friendly display format
    
    Args:
        timestamps: Raw timestamp data
        
    Returns:
        Formatted timestamps for display
    """
    try:
        # Group words into phrases/sentences for better display
        formatted = []
        current_segment = {"text": "", "start_time": None, "end_time": None}
        segment_counter = 0
        
        for i, item in enumerate(timestamps):
            if not isinstance(item, dict):
                continue
                
            word = item.get("word", "")
            start_time = item.get("start_time", 0)
            end_time = item.get("end_time", 0)
            
            # Start a new segment if this is the first word
            if current_segment["start_time"] is None:
                current_segment["start_time"] = start_time
                current_segment["text"] = word
            else:
                # Add space before word unless it's punctuation
                if word and not word.startswith((',', '.', '!', '?', ':', ';')):
                    current_segment["text"] += " "
                current_segment["text"] += word
            
            current_segment["end_time"] = end_time
            
            # Create a new segment every ~10-15 words or on sentence end
            if (i > 0 and (i+1) % 10 == 0) or (word and word.endswith(('.', '!', '?'))):
                # Format times as MM:SS
                start_formatted = format_seconds_to_mmss(current_segment["start_time"])
                end_formatted = format_seconds_to_mmss(current_segment["end_time"])
                
                formatted.append({
                    "text": current_segment["text"].strip(),
                    "start_time": current_segment["start_time"],
                    "end_time": current_segment["end_time"],
                    "display_time": f"{start_formatted} - {end_formatted}"
                })
                
                # Reset for next segment
                segment_counter += 1
                current_segment = {"text": "", "start_time": None, "end_time": None}
        
        # Add the last segment if there's any content
        if current_segment["text"]:
            start_formatted = format_seconds_to_mmss(current_segment["start_time"])
            end_formatted = format_seconds_to_mmss(current_segment["end_time"])
            
            formatted.append({
                "text": current_segment["text"].strip(),
                "start_time": current_segment["start_time"],
                "end_time": current_segment["end_time"],
                "display_time": f"{start_formatted} - {end_formatted}"
            })
        
        return formatted
    
    except Exception as e:
        logger.error(f"Error formatting timestamps: {str(e)}")
        return []

def format_seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    if seconds is None:
        return "00:00"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# Clear AI cache periodically (e.g., every hour)
async def clear_ai_cache_periodically():
    """Periodically clear the AI response cache to prevent memory buildup"""
    while True:
        try:
            await asyncio.sleep(3600)  # 1 hour
            AI_CACHE.clear()
            logger.info("AI response cache cleared")
        except Exception as e:
            logger.error(f"Error clearing AI cache: {str(e)}")
            await asyncio.sleep(60)  # Retry after 1 minute

# Start cache clearing task
try:
    asyncio.create_task(clear_ai_cache_periodically())
except Exception as e:
    logger.error(f"Failed to start cache clearing task: {str(e)}")