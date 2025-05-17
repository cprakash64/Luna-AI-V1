"""
CORS middleware for Luna AI
Ensures all routes, including Socket.IO, have proper CORS headers
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class AllCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom middleware that adds permissive CORS headers to every response
    This is meant for development only
    """
    
    def __init__(self, app, allowed_origins: Union[List[str], str] = "*"):
        super().__init__(app)
        self.allowed_origins = allowed_origins
    
    async def dispatch(self, request, call_next):
        # Log request for debugging
        logger.debug(f"CORS middleware handling: {request.method} {request.url.path}")
        
        origin = request.headers.get("origin")
        
        # Determine if the origin is allowed
        if self.allowed_origins == "*":
            allow_origin = "*"
        elif origin and origin in self.allowed_origins:
            allow_origin = origin
        else:
            allow_origin = self.allowed_origins[0] if self.allowed_origins else "*"
        
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            logger.info(f"Handling CORS preflight request for {request.url.path} from origin {origin}")
            
            response = Response(
                content="",
                status_code=200,
            )
            
            # Add CORS headers
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
            response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
            
            # Only add credentials header if not using wildcard origin
            if allow_origin != "*":
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response
        
        # Process the request normally
        try:
            response = await call_next(request)
            
            # Add CORS headers to all responses
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
            
            # Only add credentials header if not using wildcard origin
            if allow_origin != "*":
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response
        except Exception as e:
            logger.error(f"Error during request handling: {str(e)}")
            response = Response(content=str(e), status_code=500)
            
            # Add CORS headers to error responses too
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
            
            if allow_origin != "*":
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response

def setup_cors(app, allowed_origins=None):
    """
    Register the CORS middleware with the application
    
    Args:
        app: The FastAPI application
        allowed_origins: List of allowed origins or "*" for all origins
    """
    if allowed_origins is None:
        # Default to specific origins for development
        allowed_origins = [
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
            "http://localhost:3000",  # Common React dev server port
            "http://127.0.0.1:3000",
            "http://localhost:8080",  # Common alternative port
            "http://127.0.0.1:8080",
            "http://localhost:4173",  # Vite preview server
            "http://127.0.0.1:4173"
        ]
    
    logger.info(f"Setting up CORS middleware with allowed origins: {allowed_origins}")
    app.add_middleware(AllCORSMiddleware, allowed_origins=allowed_origins)