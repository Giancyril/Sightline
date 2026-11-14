"""
api/tests/test_detect.py
------------------------
Stage 6 unit and integration tests for FastAPI inference endpoints.
Covers:
  - GET /v1/classes (20-class taxonomy, colors, metadata)
  - GET /v1/model (architecture, input spec, benchmark metrics)
  - POST /v1/detect (JPEG, PNG, invalid format, empty payload, 10MB limit, thresholds)
  - DetectorService latency (<300ms SLA)
"""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, Response, UploadFile
from PIL import Image

from api.main import app
from api.routers.classes import get_classes
from api.routers.detect import detect_objects
from api.routers.model import get_model_info
from api.schemas import ClassesResponse, DetectionResponse, ModelInfoResponse
from api.services.detector import (
    MAX_FILE_SIZE_BYTES,
    DetectorService,
    ImageValidationError,
    get_detector,
)

pytestmark = pytest.mark.stage6


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def create_test_image(
    width: int = 320,
    height: int = 240,
    fmt: str = "JPEG",
    color: tuple[int, int, int] = (120, 180, 220),
) -> bytes:
    """Generate in-memory test image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_upload_file(data: bytes, filename: str = "test.jpg") -> UploadFile:
    """Create a FastAPI UploadFile wrapper around bytes."""
    return UploadFile(filename=filename, file=io.BytesIO(data))


# ---------------------------------------------------------------------------
# Routing Table Registration Tests
# ---------------------------------------------------------------------------


def test_routes_registered():
    """Verify all required endpoints are mounted in the FastAPI routing table."""
    paths = list(app.openapi()["paths"].keys())
    assert "/v1/detect" in paths, "Missing POST /v1/detect route"
    assert "/v1/classes" in paths, "Missing GET /v1/classes route"
    assert "/v1/model" in paths, "Missing GET /v1/model route"
    assert "/healthz" in paths, "Missing GET /healthz route"


# ---------------------------------------------------------------------------
# GET /v1/classes Tests
# ---------------------------------------------------------------------------


def test_get_classes_taxonomy():
    """Verify the 20-class taxonomy returns complete and valid metadata."""
    response = get_classes()
    assert isinstance(response, ClassesResponse)
    assert response.total_classes == 20
    assert len(response.classes) == 20

    ids = [c.id for c in response.classes]
    assert ids == list(range(1, 21)), "Class IDs must be 1 to 20 continuous"

    for c in response.classes:
        assert len(c.name) > 0, "Class name must not be empty"
        assert c.mid.startswith("/m/"), f"Invalid Open Images MID: {c.mid}"
        assert c.color_hex.startswith("#") and len(c.color_hex) == 7
        assert len(c.color_rgb) == 3
        assert all(0 <= v <= 255 for v in c.color_rgb)
        assert len(c.description) > 5

    # Check shared vs novel split
    coco_shared = [c for c in response.classes if c.is_coco]
    novel_classes = [c for c in response.classes if not c.is_coco]
    assert len(coco_shared) == 10, "Expected 10 COCO-overlapping classes"
    assert len(novel_classes) == 10, "Expected 10 novel classes"


# ---------------------------------------------------------------------------
# GET /v1/model Tests
# ---------------------------------------------------------------------------


def test_get_model_metadata():
    """Verify model specification and benchmark metrics."""
    response = get_model_info()
    assert isinstance(response, ModelInfoResponse)
    assert response.version == "v1.0.0"
    assert response.architecture == "SSDMobileNetV2"
    assert response.input_resolution == [300, 300]
    assert response.num_classes == 21
    assert response.optimal_confidence_threshold == 0.25
    assert "JPEG" in response.supported_formats
    assert "PNG" in response.supported_formats
    assert "WebP" in response.supported_formats

    # Accuracy metrics
    assert response.metrics.mAP_all > 0.20
    assert response.metrics.mAP_novel_classes > 0.10
    assert response.metrics.mAP_coco_shared > 0.30


# ---------------------------------------------------------------------------
# POST /v1/detect Tests
# ---------------------------------------------------------------------------


def test_detect_jpeg_success():
    """Test object detection with a standard JPEG image."""
    data = create_test_image(width=320, height=240, fmt="JPEG")
    # Warmup detector to eliminate cold-start graph initialization overhead
    get_detector().detect(data)

    upload = make_upload_file(data, "photo.jpg")

    response = detect_objects(
        response=Response(),
        file=upload,
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    assert isinstance(response, DetectionResponse)
    assert response.image_dimensions.width == 320
    assert response.image_dimensions.height == 240
    assert response.image_dimensions.channels == 3
    assert response.model_version == "v1.0.0"
    assert response.inference_time_ms > 0.0
    assert response.inference_time_ms < 300.0, "Inference must meet <300ms SLA"

    # Verify detection structure
    for det in response.detections:
        assert 1 <= det.class_id <= 20
        assert len(det.class_name) > 0
        assert 0.0 <= det.confidence <= 1.0
        assert det.confidence >= 0.25

        # Normalized coordinates [0, 1]
        assert 0.0 <= det.box_normalized.ymin <= 1.0
        assert 0.0 <= det.box_normalized.xmin <= 1.0
        assert 0.0 <= det.box_normalized.ymax <= 1.0
        assert 0.0 <= det.box_normalized.xmax <= 1.0
        assert det.box_normalized.ymax >= det.box_normalized.ymin
        assert det.box_normalized.xmax >= det.box_normalized.xmin

        # Pixel coordinates
        assert 0 <= det.box_pixel.ymin <= 240
        assert 0 <= det.box_pixel.xmin <= 320
        assert 0 <= det.box_pixel.ymax <= 240
        assert 0 <= det.box_pixel.xmax <= 320
        assert det.box_pixel.width >= 0
        assert det.box_pixel.height >= 0

        # UI Color
        assert det.color_hex.startswith("#")
        assert len(det.color_rgb) == 3


def test_detect_png_success():
    """Test object detection with a PNG image."""
    data = create_test_image(width=150, height=150, fmt="PNG")
    upload = make_upload_file(data, "icon.png")

    response = detect_objects(
        response=Response(), file=upload, confidence_threshold=0.20
    )  # type: ignore[arg-type]
    assert isinstance(response, DetectionResponse)
    assert response.image_dimensions.width == 150
    assert response.image_dimensions.height == 150


def test_detect_empty_payload_raises_400():
    """Empty payload must raise 400 Bad Request."""
    upload = make_upload_file(b"", "empty.jpg")
    with pytest.raises(HTTPException) as exc_info:
        detect_objects(response=Response(), file=upload)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 400
    assert "Empty image" in exc_info.value.detail


def test_detect_unsupported_format_raises_415():
    """Arbitrary text or non-image files must raise 415 Unsupported Media Type."""
    text_data = b"Hello, this is a plain text file, not an image!"
    upload = make_upload_file(text_data, "document.txt")
    with pytest.raises(HTTPException) as exc_info:
        detect_objects(response=Response(), file=upload)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 415
    assert "Unsupported file format" in exc_info.value.detail


def test_detect_oversized_payload_raises_413():
    """Files exceeding 10MB must raise 413 Payload Too Large."""
    detector = get_detector()
    oversized = b"\xff\xd8\xff" + b"0" * (MAX_FILE_SIZE_BYTES + 1024)

    with pytest.raises(ImageValidationError) as exc_info:
        detector.validate_image_bytes(oversized)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_detect_high_confidence_threshold():
    """Applying high confidence threshold should filter detections."""
    data = create_test_image(width=300, height=300, fmt="JPEG")
    upload = make_upload_file(data, "photo.jpg")

    response = detect_objects(
        response=Response(), file=upload, confidence_threshold=0.999
    )  # type: ignore[arg-type]
    assert isinstance(response, DetectionResponse)
    for det in response.detections:
        assert det.confidence >= 0.999


# ---------------------------------------------------------------------------
# Detector Service Tests
# ---------------------------------------------------------------------------


def test_detector_service_latency():
    """Verify pure model forward pass execution time is < 300 ms SLA."""
    service = DetectorService()
    img_data = create_test_image(width=300, height=300, fmt="JPEG")

    result = service.detect(img_data)
    assert result["inference_time_ms"] < 300.0, (
        f"Inference latency {result['inference_time_ms']} ms exceeds 300 ms SLA"
    )


# ---------------------------------------------------------------------------
# Commit 4: model_id query param on POST /v1/detect
# ---------------------------------------------------------------------------


def test_detect_default_model_id():
    """When no model_id supplied the fine-tuned model is used."""
    data = create_test_image(width=300, height=300, fmt="JPEG")
    upload = make_upload_file(data, "photo.jpg")

    response = detect_objects(
        response=Response(),
        file=upload,
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    assert isinstance(response, DetectionResponse)
    assert response.model_id == "fine-tuned"


def test_detect_explicit_model_id_fine_tuned():
    """Explicitly passing model_id='fine-tuned' returns the same model."""
    data = create_test_image(width=300, height=300, fmt="JPEG")
    upload = make_upload_file(data, "photo.jpg")

    response = detect_objects(
        response=Response(),
        file=upload,
        model_id="fine-tuned",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    assert isinstance(response, DetectionResponse)
    assert response.model_id == "fine-tuned"
    assert response.model_version == "v1.0.0"


def test_detect_unknown_model_id_raises_404():
    """An unknown model_id must raise HTTPException 404."""
    data = create_test_image(width=300, height=300, fmt="JPEG")
    upload = make_upload_file(data, "photo.jpg")

    with pytest.raises(HTTPException) as exc_info:
        detect_objects(
            response=Response(),
            file=upload,
            model_id="does-not-exist",
            confidence_threshold=0.25,
            nms_iou_threshold=0.45,
        )
    assert exc_info.value.status_code == 404
    assert "does-not-exist" in exc_info.value.detail


def test_detect_response_includes_model_id_and_version():
    """DetectionResponse body must carry model_id and model_version fields, and set response headers."""
    data = create_test_image(width=300, height=300, fmt="JPEG")
    upload = make_upload_file(data, "photo.jpg")
    res = Response()

    response = detect_objects(
        response=res,
        file=upload,
        model_id="fine-tuned",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    assert isinstance(response, DetectionResponse)
    assert response.model_id == "fine-tuned"
    assert response.model_version  # non-empty string
    assert response.model_name  # non-empty string
    assert res.headers.get("X-Model-Id") == "fine-tuned"
    assert res.headers.get("X-Model-Version") == "v1.0.0"
