"""
backend/routers/detection.py
-----------------------------
API routes for road defect detection and reporting.

Endpoints:
    POST /api/detect
        - Upload image/video
        - Returns: detections, severity, annotated image, report

    POST /api/report
        - Send report to municipality
        - Returns: confirmation ID

    GET /api/status
        - Check detector and model status
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from backend.utils import analyze_image, get_detector, error_response

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models for request/response
# ─────────────────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """Request body for detection"""
    location_hint: str = "Unknown Location"


class ReportRequest(BaseModel):
    """Request to send report to municipality"""
    report_id: str
    municipality_email: Optional[str] = None
    additional_notes: Optional[str] = None


class StatusResponse(BaseModel):
    """Status check response"""
    status: str
    model_loaded: bool
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Detection endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    location_hint: str = Form(default="Unknown Location")
):
    """
    Upload an image and detect road defects.
    
    Args:
        file: Image file (JPEG, PNG, WebP)
        location_hint: Human-readable location description
    
    Returns:
        JSON with:
        - detection: bounding boxes, labels, confidence scores
        - severity: score, level, recommended action
        - annotated_image: base64 encoded image with boxes
        - report: official municipal report
    """
    try:
        # Validate file type
        allowed_types = {"image/jpeg", "image/png", "image/webp"}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Use JPEG, PNG, or WebP. Got: {file.content_type}"
            )
        
        # Check file size (max 10MB)
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
                detail="File too large. Maximum size: 10MB"
            )
        
        # Check if detector is loaded
        detector = get_detector()
        if not detector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI detector not ready. Try again in a moment."
            )
        
        # Run full pipeline
        result = await analyze_image(file_bytes, location_hint)
        return result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[detect] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Report submission endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/report")
async def submit_report(req: ReportRequest):
    """
    Submit detection report to municipality.
    
    Args:
        report_id: ID of the report to submit
        municipality_email: Email address of municipal authority (optional)
        additional_notes: Any additional notes (optional)
    
    Returns:
        Confirmation with submission ID and timestamp
    """
    try:
        # In a real system, this would:
        # 1. Store the submission in a database
        # 2. Send email to municipality
        # 3. Log the transaction
        
        submission_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        print(f"[report] Submission received:")
        print(f"  Report ID: {report_id}")
        print(f"  Report ID: {req.report_id}")
        print(f"  Municipality: {req.municipality_email}")
        print(f"  Submission ID: {submission_id}")
        
        return {
            "status": "submitted",
            "submission_id": submission_id,
            "report_id": req.report_id,
            "timestamp": timestamp,
            "message": "Report successfully submitted to municipal authority. You will receive a confirmation email shortly.",
        }
    
    except Exception as e:
        print(f"[report] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Submission failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Status endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status")
async def detector_status() -> StatusResponse:
    """
    Check if detector and model are loaded and ready.
    
    Returns:
        Status of the AI detector
    """
    detector = get_detector()
    
    return StatusResponse(
        status="ready" if detector else "loading",
        model_loaded=detector is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/test")
async def test_connection():
    """Test endpoint to verify API is running"""
    return {
        "status": "connected",
        "message": "SmartRoad API is running",
        "timestamp": datetime.utcnow().isoformat(),
    }
