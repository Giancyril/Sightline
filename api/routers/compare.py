"""
api/routers/compare.py
----------------------
Endpoint for comparing two object detection models in parallel on a single image.
"""

from __future__ import annotations

import concurrent.futures
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from api.schemas import CompareResponse, DetectionResponse
from api.services.detector import ImageValidationError, UnknownModelError, get_detector

router = APIRouter(prefix="/v1", tags=["Model Comparison"])


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two models in parallel on an uploaded image",
    description=(
        "Upload an image and run two registered models concurrently. "
        "Returns detection outputs and per-model latency for both models."
    ),
    responses={
        200: {"description": "Successful comparison result."},
        400: {"description": "Corrupt image or bad payload."},
        404: {"description": "One or both model IDs not found in registry."},
        413: {"description": "Payload too large (>10MB)."},
        415: {"description": "Unsupported image format."},
    },
)
def compare_models(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)"),
    model_a: str = Query(
        default="fine-tuned",
        description="Registry ID for the first model",
    ),
    model_b: str = Query(
        default="coco-baseline",
        description="Registry ID for the second model",
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
) -> CompareResponse:
    """Run parallel inference across two models on the same image."""
    start_total = time.perf_counter()
    try:
        content = file.file.read() if hasattr(file, "file") else file.read()

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
        resolved_a = model_a.default if hasattr(model_a, "default") else str(model_a)
        resolved_b = model_b.default if hasattr(model_b, "default") else str(model_b)

        detector_a = get_detector(resolved_a)
        detector_b = get_detector(resolved_b)

        # Validate image bytes once upfront using detector_a
        detector_a.validate_image_bytes(content)

        # Run both model inferences in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                detector_a.detect,
                content,
                conf_thresh,
                nms_thresh,
            )
            future_b = executor.submit(
                detector_b.detect,
                content,
                conf_thresh,
                nms_thresh,
            )
            res_a = future_a.result()
            res_b = future_b.result()

        total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)

        return CompareResponse(
            model_a_result=DetectionResponse(**res_a),
            model_b_result=DetectionResponse(**res_b),
            total_time_ms=total_time_ms,
        )
    except UnknownModelError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: {e}",
        ) from e
    except ImageValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison inference failure: {e}",
        ) from e
