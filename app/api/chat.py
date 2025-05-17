"""
Chat API endpoints for Luna AI
Handles chat functionality with the AI assistant
"""
from typing import Any, Dict
from uuid import UUID
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
# Import from our new simplified structure
from app.utils.database import get_db
from app.services.auth import get_current_user
# Fix: Import generate_ai_response instead of generate_chat_response
from app.services.ai import generate_ai_response
from app.models.user import User

router = APIRouter()
logger = logging.getLogger("chat_api")

@router.post("/", response_model=Dict[str, str])
async def chat_with_video(
    *,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Chat with a video or general assistant
    If video_id is provided, context from the video is used
    Otherwise, operates in general assistant mode
    """
    try:
        # Parse request body
        data = await request.json()
        message = data.get("message")
        video_id = data.get("video_id")
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is required"
            )
            
        # If video_id is provided, verify ownership
        if video_id:
            from sqlalchemy import text
            
            result = db.execute(
                text("SELECT id FROM videos WHERE id = :id AND user_id = :user_id"),
                {"id": str(video_id), "user_id": current_user.id}
            )
            
            if not result.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video not found"
                )
        
        # Fix: Use generate_ai_response instead of generate_chat_response
        response = await generate_ai_response(
            question=message,  # Update the parameter name to match generate_ai_response signature
            video_id=video_id,
            # Remove user_id as generate_ai_response doesn't take this parameter
        )
        
        # Extract the answer from the response dict returned by generate_ai_response
        answer = response.get("answer", "Sorry, I couldn't generate a response")
        
        return {"response": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )