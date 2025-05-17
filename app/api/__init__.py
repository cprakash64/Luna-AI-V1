# File: app/api/__init__.py

from fastapi import APIRouter
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.videos import router as videos_router
from app.api.websockets import router as websocket_router
from app.api.timestamps import router as timestamps_router  # Import the new timestamps router

# Try to import the visual analysis routers, with fallback handling
try:
    from app.api.videos_visual import (
        videos_router as videos_visual_router,
        videos_api_router as videos_api_visual_router,
        analysis_router as visual_analysis_router
    )
    has_visual_routers = True
except ImportError:
    # Try single router approach
    try:
        from app.api.videos_visual import router as videos_visual_router
        has_visual_routers = True
    except ImportError:
        # Log that visual analysis is not available
        import logging
        logging.warning("Visual analysis module not available. Some endpoints will return 404.")
        has_visual_routers = False

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat_router, prefix="/chat", tags=["Chat"])
api_router.include_router(videos_router, prefix="/videos", tags=["Videos"])
api_router.include_router(websocket_router, prefix="/ws", tags=["WebSockets"])
api_router.include_router(timestamps_router, tags=["Timestamps"])  # Add the new timestamps router

# Include visual analysis routers if available
if has_visual_routers:
    # If we have three separate routers
    if 'videos_api_visual_router' in locals() and 'visual_analysis_router' in locals():
        api_router.include_router(videos_visual_router, tags=["Visual Analysis"])
        api_router.include_router(videos_api_visual_router, tags=["Visual Analysis"])
        api_router.include_router(visual_analysis_router, tags=["Visual Analysis"])
    else:
        # If we have a single router
        api_router.include_router(videos_visual_router, tags=["Visual Analysis"])