"""
Visual analysis endpoints for Luna AI
Provides API routes for video visual analysis features
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, Response
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import os
import logging
import traceback
from fastapi.middleware.cors import CORSMiddleware

from app.utils.database import get_db
from app.services.visual_analysis import analyze_video_frames, create_debug_visual_data

# Configure logging
logger = logging.getLogger(__name__)

# Environment configuration
DEV_MODE = os.environ.get("DEV_MODE", "False").lower() in ("true", "1", "t")
DEBUG_MODE = os.environ.get("DEBUG_VISUAL_ANALYSIS", "False").lower() in ("true", "1", "t")
logger.info(f"Visual analysis API running in {'development' if DEV_MODE else 'production'} mode")

# Main router that will be imported by main.py
router = APIRouter()

# Create individual specialized routers
videos_router = APIRouter(prefix="/videos")
videos_api_router = APIRouter(prefix="/api")
analysis_router = APIRouter()

# Standard response template
def create_visual_response(video_id: str, status: str = "pending", data: Optional[Dict] = None):
    """Create a standardized response for visual analysis endpoints"""
    response = {
        "status": status,
        "video_id": video_id,
        "message": f"Visual analysis {status}",
    }
    
    # Add data if provided
    if data:
        response["data"] = data
    else:
        # Add placeholder data
        response["data"] = {
            "frames": [],
            "scenes": [],
            "objects": [],
            "highlights": [],
            "topics": [],
            "content_type": "video"
        }
        
    return response

# Visual data helper with proper error handling
async def get_visual_analysis_data(video_id: str) -> Dict[str, Any]:
    """Get visual analysis data with proper error handling"""
    try:
        if DEBUG_MODE:
            # Use mock data in debug mode - this is synchronous
            visual_data = create_debug_visual_data(video_id)
            return visual_data
        else:
            # This is async and needs to be awaited
            visual_data = await analyze_video_frames(video_id)
            return visual_data
    except Exception as e:
        logger.error(f"Error in get_visual_analysis_data: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty data structure rather than raising exception
        return {
            "frames": [], 
            "scenes": [], 
            "objects": [],
            "highlights": [],
            "error": str(e)
        }

# Handler for authentication in dev/prod modes
async def handle_auth(request: Request):
    """Handle authentication with dev mode support"""
    try:
        from app.services.auth import get_current_user
        
        # Get database session
        db = next(get_db())
        
        if DEV_MODE:
            try:
                # Try to get authenticated user
                user = await get_current_user(request, db=db)
                return user
            except Exception as e:
                # Provide default testing user in development
                logger.warning(f"Auth bypassed in development mode: {str(e)}")
                return {
                    "id": "dev-user",
                    "email": "dev@example.com",
                    "username": "dev_user",
                    "is_active": True,
                    "authenticated": True
                }
        else:
            # Require authentication in production
            return await get_current_user(request, db=db)
    except Exception as e:
        logger.error(f"Error in authentication: {str(e)}")
        logger.error(traceback.format_exc())
        if DEV_MODE:
            # Default user for development
            return {
                "id": "dev-user",
                "email": "dev@example.com",
                "username": "dev_user",
                "is_active": True,
                "authenticated": True
            }
        else:
            # Re-raise in production
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )

# Helper function to add CORS headers
def add_cors_headers(response: Response):
    """Add CORS headers to response"""
    response.headers["Access-Control-Allow-Origin"] = "*"  # Allow all origins in development
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Primary API endpoints

# -- Visual data endpoint
@router.get("/api/v1/videos/{video_id}/visual-data", tags=["Visual Analysis"])
async def get_video_visual_data(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Get visual data for a specific video"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Visual data requested for video: {video_id}")
    try:
        # Get visual analysis data
        visual_data = await get_visual_analysis_data(video_id)
        
        # Return the data or a placeholder if none is available
        if not visual_data:
            return create_visual_response(video_id)
        
        return create_visual_response(video_id, "success", visual_data)
    except Exception as e:
        logger.error(f"Error retrieving visual data: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

# -- API visual endpoint
@router.get("/api/videos/{video_id}/visual", tags=["Visual Analysis"])
@router.get("/api/v1/videos/{video_id}/visual", tags=["Visual Analysis"])
async def get_api_video_visual(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Get visual data for a specific video (API endpoint)"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Visual API data requested for video: {video_id}")
    try:
        # Try to authenticate user, with fallback for dev mode
        try:
            await handle_auth(request)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return create_visual_response(video_id, "error", {"error": "Authentication failed"})
        
        # Get visual data
        visual_data = await get_visual_analysis_data(video_id)
        
        # Return the data or a placeholder if none is available
        if not visual_data:
            return create_visual_response(video_id)
        
        return create_visual_response(video_id, "success", visual_data)
    except Exception as e:
        logger.error(f"Error retrieving visual API data: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

# -- Analyze visual endpoint
@router.get("/api/analyze-visual", tags=["Visual Analysis"])
async def analyze_visual_api(
    request: Request,
    response: Response,
    videoId: str = Query(None),
    db: Session = Depends(get_db)
):
    """API endpoint for analyzing visual content"""
    response = add_cors_headers(response)  # Add CORS headers
    if not videoId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="videoId is required"
        )
    
    logger.info(f"Visual analysis requested for video: {videoId}")
    try:
        # Get visual analysis data
        result = await get_visual_analysis_data(videoId)
        return create_visual_response(videoId, "success", result)
    except Exception as e:
        logger.error(f"Error analyzing visual data: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(videoId, "error", {"error": str(e)})

# YouTube specific routes
@router.get("/api/visual-analysis/youtube_{youtube_id}", tags=["Visual Analysis"])
async def get_youtube_visual_analysis(
    youtube_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Get visual analysis for YouTube videos"""
    response = add_cors_headers(response)  # Add CORS headers
    full_video_id = f"youtube_{youtube_id}"
    logger.info(f"YouTube visual analysis requested for video: {full_video_id}")
    try:
        # Get analysis data - PROPERLY AWAIT
        result = await get_visual_analysis_data(full_video_id)
        return create_visual_response(full_video_id, "success", result)
    except Exception as e:
        logger.error(f"Error getting YouTube visual analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(full_video_id, "error", {"error": str(e)})

@router.get("/api/v1/visual-analysis/youtube_{youtube_id}", tags=["Visual Analysis"])
async def get_youtube_visual_analysis_v1(
    youtube_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Get visual analysis for YouTube videos (v1 API)"""
    response = add_cors_headers(response)  # Add CORS headers
    full_video_id = f"youtube_{youtube_id}"
    logger.info(f"YouTube v1 visual analysis requested for video: {full_video_id}")
    try:
        # Get analysis data - PROPERLY AWAIT
        result = await get_visual_analysis_data(full_video_id)
        return create_visual_response(full_video_id, "success", result)
    except Exception as e:
        logger.error(f"Error getting YouTube v1 visual analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(full_video_id, "error", {"error": str(e)})

# Generic visual analysis endpoints
@router.get("/api/visual-analysis/{video_id}", tags=["Visual Analysis"])
async def get_visual_analysis_by_id(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Get visual analysis by video ID"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Visual analysis by ID requested for video: {video_id}")
    try:
        # Get analysis data with proper await
        result = await get_visual_analysis_data(video_id)
        return create_visual_response(video_id, "success", result)
    except Exception as e:
        logger.error(f"Error getting visual analysis by ID: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

@router.get("/api/v1/visual-analysis/{video_id}", tags=["Visual Analysis"])
async def get_visual_analysis_by_id_v1(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Get visual analysis by video ID (v1 API)"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Visual analysis by ID requested for video (v1): {video_id}")
    try:
        # Get analysis data with proper await
        result = await get_visual_analysis_data(video_id)
        return create_visual_response(video_id, "success", result)
    except Exception as e:
        logger.error(f"Error getting visual analysis by ID (v1): {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

# Endpoint with query parameters
@router.get("/api/v1/visual-analysis", tags=["Visual Analysis"])
async def get_api_visual_analysis(
    request: Request,
    response: Response,
    video_id: str = Query(None),
    db: Session = Depends(get_db)
):
    """Get visual analysis data with query parameters"""
    response = add_cors_headers(response)  # Add CORS headers
    if not video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="video_id is required"
        )
    
    logger.info(f"Visual analysis requested with query param for video: {video_id}")
    try:
        # Get analysis data with proper await
        result = await get_visual_analysis_data(video_id)
        return create_visual_response(video_id, "success", result)
    except Exception as e:
        logger.error(f"Error analyzing visual data with query param: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

# POST endpoint for triggering analysis
@router.post("/api/v1/videos/{video_id}/analyze-visual", tags=["Visual Analysis"])
async def trigger_visual_analysis(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Trigger visual analysis for a video"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Visual analysis POST request for video: {video_id}")
    try:
        # Get visual analysis data with proper await
        result = await get_visual_analysis_data(video_id)
        return create_visual_response(video_id, "success", result)
    except Exception as e:
        logger.error(f"Error triggering visual analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

# -- Add OPTIONS endpoints for CORS preflight requests
@router.options("/api/v1/videos/{video_id}/analyze-visual", tags=["Visual Analysis"])
@router.options("/api/v1/videos/{video_id}/visual-data", tags=["Visual Analysis"])
@router.options("/api/videos/{video_id}/visual", tags=["Visual Analysis"])
@router.options("/api/v1/videos/{video_id}/visual", tags=["Visual Analysis"])
@router.options("/api/analyze-visual", tags=["Visual Analysis"])
@router.options("/api/visual-analysis/{video_id}", tags=["Visual Analysis"])
@router.options("/api/v1/visual-analysis/{video_id}", tags=["Visual Analysis"])
@router.options("/api/v1/visual-analysis", tags=["Visual Analysis"])
async def options_handler(response: Response):
    """Handle OPTIONS preflight requests for CORS"""
    return add_cors_headers(response)

# Mount legacy routers to the main router with prefix
router.include_router(videos_router, prefix="/videos")
router.include_router(videos_api_router, prefix="/api")
router.include_router(analysis_router, prefix="/analysis")

# Legacy routes used by older code - now properly mounted
@videos_router.get("/{video_id}/visual", tags=["Visual Analysis"])
async def get_video_visual_legacy(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Legacy endpoint for visual data"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Legacy visual data requested for video: {video_id}")
    try:
        # Get visual analysis data with proper await
        visual_data = await get_visual_analysis_data(video_id)
        
        # Return the data or a placeholder if none is available
        if not visual_data:
            return create_visual_response(video_id)
        
        return create_visual_response(video_id, "success", visual_data)
    except Exception as e:
        logger.error(f"Error in legacy visual endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

@videos_api_router.get("/videos/{video_id}/visual", tags=["Visual Analysis"])
async def get_api_video_visual_legacy(
    video_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Legacy API endpoint for visual data"""
    response = add_cors_headers(response)  # Add CORS headers
    logger.info(f"Legacy API visual data requested for video: {video_id}")
    try:
        # Get visual analysis data with proper await
        visual_data = await get_visual_analysis_data(video_id)
        
        # Return the data or a placeholder if none is available
        if not visual_data:
            return create_visual_response(video_id)
        
        return create_visual_response(video_id, "success", visual_data)
    except Exception as e:
        logger.error(f"Error in legacy API visual endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(video_id, "error", {"error": str(e)})

@analysis_router.get("/analyze-visual", tags=["Visual Analysis"])
async def analyze_visual_legacy(
    request: Request,
    response: Response,
    videoId: str = Query(None),
    db: Session = Depends(get_db)
):
    """Legacy endpoint for visual analysis"""
    response = add_cors_headers(response)  # Add CORS headers
    if not videoId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="videoId is required"
        )
    
    logger.info(f"Legacy visual analysis requested for video: {videoId}")
    try:
        # Get visual analysis data with proper await
        result = await get_visual_analysis_data(videoId)
        return create_visual_response(videoId, "success", result)
    except Exception as e:
        logger.error(f"Error in legacy analyze endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return create_visual_response(videoId, "error", {"error": str(e)})

# Health check endpoint
@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "visual_analysis"}

# Add a configuration function for CORS in main.py
def setup_cors(app):
    """Setup CORS middleware for the application"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Your frontend URL
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )