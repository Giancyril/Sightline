"""
api/routers/detect.py
---------------------
Inference endpoint for single-image and webcam frame object detection.
Accepts multipart file uploads with optional threshold parameters.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from api.schemas import DetectionResponse
from api.services.detector import ImageValidationError, get_detector

router = APIRouter(prefix="/v1", tags=["Detection"])


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect objects in an uploaded image or video frame",
    description=(
        "Upload a JPEG, PNG, or WebP image up to 10MB. "
        "Returns normalized bounding boxes, pixel coordinates, class labels, and confidence scores."
    ),
    responses={
        200: {"description": "Successful object detection result."},
        400: {"description": "Corrupt image or bad payload."},
        413: {"description": "Payload too large (>10MB)."},
        415: {"description": "Unsupported image format."},
    },
)
def detect_objects(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)"),
    confidence_threshold: float = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence score threshold for reported detections",
    ),
    nms_iou_threshold: float = Query(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="Non-Maximum Suppression (NMS) intersection-over-union threshold",
    ),
) -> DetectionResponse:
    """Run object detection on uploaded image."""
    try:
        content = file.file.read() if hasattr(file, "file") else file.read()

        # Handle direct function calls where Query parameter defaults are passed
        conf_thresh = (
            float(confidence_threshold.default)
            if hasattr(confidence_threshold, "default")
            else float(confidence_threshold)
        )
        nms_thresh = (
            float(nms_iou_threshold.default)
            if hasattr(nms_iou_threshold, "default")
            else float(nms_iou_threshold)
        )

        detector = get_detector()
        result = detector.detect(
            image_bytes=content,
            confidence_threshold=conf_thresh,
            nms_iou_threshold=nms_thresh,
        )
        return DetectionResponse(**result)
    except ImageValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference failure: {e}",
        ) from e
