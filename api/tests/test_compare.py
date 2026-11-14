"""
api/tests/test_compare.py
-------------------------
Unit and integration tests for POST /v1/compare endpoint.
Covers:
  - Route registration in OpenAPI schema
  - Parallel inference execution across two models
  - Comparison with same model on both sides
  - 404 for unknown model_a and model_b
  - 400 on empty image payload
  - 415 on unsupported media format
  - Confidence threshold filtering on both models
"""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from api.main import app
from api.routers.compare import compare_models
from api.schemas import CompareResponse
from api.services.detector import get_detector


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


def test_compare_route_registered():
    """Verify POST /v1/compare is mounted in the FastAPI routing table."""
    paths = list(app.openapi()["paths"].keys())
    assert "/v1/compare" in paths, "Missing POST /v1/compare route"


def test_compare_success():
    """Test parallel comparison of fine-tuned and coco-baseline models."""
    data = create_test_image(width=300, height=300, fmt="JPEG")
    # Warmup both models so the comparison runs cleanly
    get_detector("fine-tuned").detect(data)
    get_detector("coco-baseline").detect(data)

    upload = make_upload_file(data, "test.jpg")
    response = compare_models(
        file=upload,
        model_a="fine-tuned",
        model_b="coco-baseline",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )

    assert isinstance(response, CompareResponse)
    assert response.total_time_ms > 0.0

    # Model A assertions
    assert response.model_a_result.model_id == "fine-tuned"
    assert response.model_a_result.model_version == "v1.0.0"
    assert response.model_a_result.inference_time_ms > 0.0
    assert response.model_a_result.image_dimensions.width == 300
    assert response.model_a_result.image_dimensions.height == 300

    # Model B assertions
    assert response.model_b_result.model_id == "coco-baseline"
    assert response.model_b_result.model_version == "v0.9.0"
    assert response.model_b_result.inference_time_ms > 0.0
    assert response.model_b_result.image_dimensions.width == 300
    assert response.model_b_result.image_dimensions.height == 300


def test_compare_same_model():
    """Running comparison with the same model on both sides succeeds."""
    data = create_test_image(width=200, height=200, fmt="JPEG")
    upload = make_upload_file(data, "same.jpg")

    response = compare_models(
        file=upload,
        model_a="fine-tuned",
        model_b="fine-tuned",
        confidence_threshold=0.30,
        nms_iou_threshold=0.45,
    )

    assert isinstance(response, CompareResponse)
    assert response.model_a_result.model_id == "fine-tuned"
    assert response.model_b_result.model_id == "fine-tuned"


def test_compare_unknown_model_a_raises_404():
    """Unknown model_a raises 404 HTTPException."""
    data = create_test_image(width=200, height=200, fmt="JPEG")
    upload = make_upload_file(data, "test.jpg")

    with pytest.raises(HTTPException) as exc_info:
        compare_models(
            file=upload,
            model_a="unknown-model-xyz",
            model_b="coco-baseline",
        )
    assert exc_info.value.status_code == 404
    assert "unknown-model-xyz" in exc_info.value.detail


def test_compare_unknown_model_b_raises_404():
    """Unknown model_b raises 404 HTTPException."""
    data = create_test_image(width=200, height=200, fmt="JPEG")
    upload = make_upload_file(data, "test.jpg")

    with pytest.raises(HTTPException) as exc_info:
        compare_models(
            file=upload,
            model_a="fine-tuned",
            model_b="unknown-model-abc",
        )
    assert exc_info.value.status_code == 404
    assert "unknown-model-abc" in exc_info.value.detail


def test_compare_empty_payload_raises_400():
    """Empty payload raises 400 Bad Request."""
    upload = make_upload_file(b"", "empty.jpg")

    with pytest.raises(HTTPException) as exc_info:
        compare_models(file=upload)
    assert exc_info.value.status_code == 400
    assert "Empty image" in exc_info.value.detail


def test_compare_unsupported_format_raises_415():
    """Non-image bytes raises 415 Unsupported Media Type."""
    upload = make_upload_file(b"just text data", "file.txt")

    with pytest.raises(HTTPException) as exc_info:
        compare_models(file=upload)
    assert exc_info.value.status_code == 415
    assert "Unsupported file format" in exc_info.value.detail
