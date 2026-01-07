"""
FastAPI Main Application

Production-grade REST API and WebSocket server for fall detection.
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio

from .session_manager import SessionManager
from .websocket_handler import WebSocketHandler
from ..utils.config import settings
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


# Global state
session_manager: SessionManager = None
websocket_handler: WebSocketHandler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global session_manager, websocket_handler
    
    logger.info("Starting Fall Detection API...")
    
    # Initialize session manager
    model_path = settings.model_path
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    session_manager = SessionManager(
        model_path=model_path,
        device=settings.device
    )
    
    # Initialize WebSocket handler
    websocket_handler = WebSocketHandler(session_manager)
    
    # Start cleanup task
    async def cleanup_task():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            session_manager.cleanup_inactive_sessions()
    
    cleanup_task_handle = asyncio.create_task(cleanup_task())
    
    logger.info("Fall Detection API started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fall Detection API...")
    cleanup_task_handle.cancel()


# Create FastAPI app
app = FastAPI(
    title="Fall Detection API",
    description="Production-grade fall detection system with real-time WebSocket streaming",
    version="1.0.0",
    lifespan=lifespan
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fall Detection API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": session_manager is not None,
        "device": settings.device,
        "active_sessions": len(session_manager.sessions) if session_manager else 0
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = session_manager.get_stats()
    return JSONResponse(content=stats)


# Session management endpoints
@app.post("/session/create")
async def create_session():
    """
    Create a new detection session.
    
    Returns:
        Session ID
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    session_id = session_manager.create_session()
    
    return {
        "session_id": session_id,
        "websocket_url": f"/ws/{session_id}"
    }


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "frames_processed": session.frames_processed,
        "detections_count": session.detections_count,
        "fall_count": session.fall_count,
        "alert_triggered": session.alert_triggered
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.remove_session(session_id)
    
    return {"message": "Session deleted successfully"}


# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time fall detection.
    
    Protocol:
    - Client sends: {"type": "frame", "frame": "<base64_image>", "timestamp": <unix_timestamp>}
    - Server sends: {"type": "result", "fall_detected": <bool>, "confidence": <float>, ...}
    - Server sends: {"type": "alert", "alert_message": "<message>"}
    - Client responds: {"type": "alert_response", "response": "ok" | "help"}
    """
    if not websocket_handler:
        await websocket.close(code=1011, reason="Service not initialized")
        return
    
    await websocket_handler.handle_connection(websocket, session_id)


# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get system configuration."""
    return {
        "fall_threshold": settings.fall_threshold,
        "confirmation_seconds": settings.confirmation_seconds,
        "alert_timeout": settings.alert_timeout,
        "target_fps": settings.target_fps,
        "buffer_size": settings.buffer_size,
        "device": settings.device
    }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Uncaught exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
        log_level=settings.log_level.lower()
    )
