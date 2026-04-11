"""
backend/main.py
---------------
Main FastAPI application for SmartRoad — AI-Powered Street Defect Detection.

Startup:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Production:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path so we can import src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the detection router
from backend.routers import detection
from backend.utils import initialize_detector, detector_cache

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan context manager: load YOLO model once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for app startup and shutdown.
    Loads the YOLO detector model once at startup.
    """
    print("[SmartRoad] Loading YOLO detector model at startup...")
    try:
        initialize_detector()
        print("[SmartRoad] ✓ Model loaded successfully")
    except Exception as e:
        print(f"[SmartRoad] ✗ Failed to load model: {e}")
        raise
    
    yield  # App runs here
    
    # Cleanup on shutdown
    print("[SmartRoad] Shutting down...")
    detector_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Create FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartRoad API",
    description="AI-Powered Street Defect Detection & Reporting for Morocco",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORS middleware — allow frontend to make requests
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Frontend dev server (if used)
        "http://localhost:8000",      # FastAPI dev server (same origin)
        "http://127.0.0.1:8000",      # Localhost alternative
        "http://127.0.0.1:3000",
        "*",  # In production, restrict this to specific domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Include routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(detection.router, prefix="/api", tags=["detection"])


# ─────────────────────────────────────────────────────────────────────────────
# Static files and frontend
# ─────────────────────────────────────────────────────────────────────────────

# Mount the frontend directory for serving index.html and assets
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

# Mount static uploads directory
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Root endpoint — serve index.html
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """
    Serve the frontend index.html
    """
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        return {"error": "Frontend not found", "path": str(index_path)}
    
    with open(index_path, "r", encoding="utf-8") as f:
        return {"html": f.read()}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "ok",
        "model_loaded": detector_cache.get("detector") is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch all exceptions and return meaningful error response"""
    return {
        "error": str(exc),
        "type": type(exc).__name__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
