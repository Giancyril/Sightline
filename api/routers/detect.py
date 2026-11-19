"""
api/routers/detect.py
---------------------
Inference endpoint for single-image and webcam frame object detection.
Accepts multipart file uploads with optional threshold parameters.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from api.schemas import DetectionResponse
from api.services.detector import ImageValidationError, UnknownModelError, get_detector

router = APIRouter(prefix="/v1", tags=["Detection"])


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect objects in an uploaded image or video frame",
    description=(
        "Upload a JPEG, PNG, or WebP image up to 10MB. "
        "Returns normalized bounding boxes, pixel coordinates, class labels, and confidence scores. "
        "Pass `model_id` to select which registered model performs inference. "
        "Pass `threshold_mode` to select 'global' (default), 'recommended' (calibrated per-class), or 'custom'."
    ),
    responses={
        200: {"description": "Successful object detection result."},
        400: {"description": "Corrupt image or bad payload."},
        404: {"description": "Unknown model_id."},
        413: {"description": "Payload too large (>10MB)."},
        415: {"description": "Unsupported image format."},
    },
)
def detect_objects(
    response: Response,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)"),
    model_id: str = Query(
        default="fine-tuned",
        description="Registry ID of the model to use for inference",
    ),
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
    threshold_mode: str = Query(
        default="global",
        description="Threshold selection mode: 'global' (default), 'recommended', or 'custom'",
    ),
    custom_thresholds: str | None = Query(
        default=None,
        description="Optional JSON string mapping class names to thresholds for 'custom' mode",
    ),
) -> DetectionResponse:
    """Run object detection on uploaded image using the specified model."""
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
        resolved_model_id = (
            model_id.default if hasattr(model_id, "default") else str(model_id)
        )
        mode_val = (
            threshold_mode.default
            if hasattr(threshold_mode, "default")
            else str(threshold_mode)
        ).lower()

        if mode_val not in ("global", "recommended", "custom"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid threshold_mode '{mode_val}'. Must be 'global', 'recommended', or 'custom'.",
            )

        parsed_custom: dict[str, float] | None = None
        if mode_val == "custom":
            raw_custom = (
                custom_thresholds.default
                if hasattr(custom_thresholds, "default")
                else custom_thresholds
            )
            if not raw_custom:
                raise HTTPException(
                    status_code=400,
                    detail="custom_thresholds JSON string is required when threshold_mode is 'custom'.",
                )
            try:
                parsed = json.loads(raw_custom)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "custom_thresholds must be a JSON object mapping class names to numbers."
                    )
                parsed_custom = {}
                for k, v in parsed.items():
                    val = float(v)
                    if not (0.0 <= val <= 1.0):
                        raise ValueError(
                            f"Threshold for '{k}' must be between 0.0 and 1.0, got {val}"
                        )
                    parsed_custom[str(k)] = val
            except Exception as err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid custom_thresholds: {err}",
                ) from err

        detector = get_detector(resolved_model_id)
        result = detector.detect(
            image_bytes=content,
            confidence_threshold=conf_thresh,
            nms_iou_threshold=nms_thresh,
            threshold_mode=mode_val,
            custom_thresholds=parsed_custom,
        )

        # Expose model identity in response headers
        response.headers["X-Model-Id"] = result.get("model_id", resolved_model_id)
        response.headers["X-Model-Version"] = result.get("model_version", "")

        return DetectionResponse(**result)
    except UnknownModelError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: {e}",
        ) from e
    except ImageValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference failure: {e}",
        ) from e
