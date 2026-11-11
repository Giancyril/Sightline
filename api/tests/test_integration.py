"""
api/tests/test_integration.py
-----------------------------
Stage 8 integration tests:
  - End-to-end detection across multiple aspect ratios & resolutions
  - API schema contract parity with TypeScript frontend definitions
  - Concurrent multi-threaded load test & warm p95 latency SLA validation (<300ms)
  - Error status code contracts (400, 413, 415)
  - Complete 20-class taxonomy contract integrity
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from api.main import app
from api.routers.classes import get_classes
from api.routers.detect import detect_objects
from api.routers.model import get_model_info
from api.schemas import DetectionResponse
from api.services.detector import MAX_FILE_SIZE_BYTES, get_detector

pytestmark = pytest.mark.stage8


def generate_image_bytes(
    width: int,
    height: int,
    fmt: str = "JPEG",
    color: tuple[int, int, int] = (100, 150, 200),
) -> bytes:
    """Generate in-memory test image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Multi-Resolution & Aspect Ratio End-to-End Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width,height",
    [
        (640, 480),  # Standard 4:3
        (1280, 720),  # 720p 16:9
        (500, 500),  # Square
        (400, 800),  # Portrait 1:2
        (800, 400),  # Landscape 2:1
    ],
)
def test_e2e_multi_aspect_ratio_detection(width: int, height: int):
    """Verify inference handles arbitrary aspect ratios with accurate scaling."""
    img_bytes = generate_image_bytes(width, height)
    upload = UploadFile(
        filename=f"test_{width}x{height}.jpg", file=io.BytesIO(img_bytes)
    )

    response = detect_objects(
        file=upload, confidence_threshold=0.25, nms_iou_threshold=0.45
    )

    assert isinstance(response, DetectionResponse)
    assert response.image_dimensions.width == width
    assert response.image_dimensions.height == height
    assert response.inference_time_ms > 0.0, "Latency must be positive"
    assert response.inference_time_ms < 1000.0, (
        "Latency must be within reasonable bounds"
    )

    for det in response.detections:
        # Pixel coordinates must strictly stay within [0, width] and [0, height]
        assert 0 <= det.box_pixel.xmin <= width
        assert 0 <= det.box_pixel.xmax <= width
        assert 0 <= det.box_pixel.ymin <= height
        assert 0 <= det.box_pixel.ymax <= height
        assert det.box_pixel.width == det.box_pixel.xmax - det.box_pixel.xmin
        assert det.box_pixel.height == det.box_pixel.ymax - det.box_pixel.ymin


# ---------------------------------------------------------------------------
# 2. Schema Contract Parity Tests
# ---------------------------------------------------------------------------


def test_schema_contract_parity():
    """Verify backend OpenAPI schema matches frontend TypeScript contracts."""
    openapi = app.openapi()
    schemas: dict[str, Any] = openapi.get("components", {}).get("schemas", {})

    # Check DetectionResponse schema
    assert "DetectionResponse" in schemas
    det_resp_fields = schemas["DetectionResponse"]["properties"]
    expected_resp_fields = [
        "model_name",
        "model_version",
        "total_detections",
        "inference_time_ms",
        "total_time_ms",
        "image_dimensions",
        "thresholds_applied",
        "detections",
    ]
    for field in expected_resp_fields:
        assert field in det_resp_fields, (
            f"Missing field '{field}' in DetectionResponse schema"
        )

    # Check DetectionItem schema
    assert "DetectionItem" in schemas
    item_fields = schemas["DetectionItem"]["properties"]
    expected_item_fields = [
        "class_id",
        "class_name",
        "confidence",
        "box_normalized",
        "box_pixel",
        "color_hex",
        "color_rgb",
    ]
    for field in expected_item_fields:
        assert field in item_fields, f"Missing field '{field}' in DetectionItem schema"

    # Check NormalizedBBox schema
    assert "NormalizedBBox" in schemas
    bbox_fields = schemas["NormalizedBBox"]["properties"]
    for coord in ["ymin", "xmin", "ymax", "xmax"]:
        assert coord in bbox_fields


# ---------------------------------------------------------------------------
# 3. Inference Latency SLA (<300ms) & Concurrency Load Test
# ---------------------------------------------------------------------------


def test_inference_latency_p95_sla():
    """Verify warm p95 inference latency < 300ms SLA (after JIT warmup)."""
    detector = get_detector()
    img_data = generate_image_bytes(320, 240)

    # Warmup: 2 discarded runs to prime JIT and memory buffers
    for _ in range(2):
        detector.detect(img_data)

    # Measure 8 warm inferences
    latencies: list[float] = []
    for _ in range(8):
        res = detector.detect(img_data, confidence_threshold=0.25)
        latencies.append(float(res["inference_time_ms"]))

    p95_latency = float(np.percentile(latencies, 95))
    mean_latency = float(np.mean(latencies))

    print(f"\n[SLA Benchmark] Mean: {mean_latency:.2f}ms | p95: {p95_latency:.2f}ms")
    assert p95_latency < 300.0, (
        f"Warm p95 inference latency of {p95_latency:.2f} ms exceeds the 300 ms SLA"
    )


def test_concurrent_inference_thread_safety():
    """Verify concurrent multi-threaded inference requests complete safely without deadlocks."""
    detector = get_detector()
    img_data = generate_image_bytes(320, 240)

    num_requests = 4
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(detector.detect, img_data, 0.25)
            for _ in range(num_requests)
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == num_requests
    for res in results:
        assert "detections" in res
        assert "inference_time_ms" in res
        assert res["total_detections"] >= 0


# ---------------------------------------------------------------------------
# 4. Error Status Code Contracts
# ---------------------------------------------------------------------------


def test_error_status_contracts():
    """Verify HTTP status codes match API contract specification."""
    # 400 Bad Request on empty file
    upload_empty = UploadFile(filename="empty.jpg", file=io.BytesIO(b""))
    with pytest.raises(HTTPException) as exc400:
        detect_objects(file=upload_empty)
    assert exc400.value.status_code == 400

    # 415 Unsupported Media Type on non-image data
    upload_text = UploadFile(
        filename="test.txt", file=io.BytesIO(b"Plain text file content")
    )
    with pytest.raises(HTTPException) as exc415:
        detect_objects(file=upload_text)
    assert exc415.value.status_code == 415

    # 413 Payload Too Large on files > 10MB
    detector = get_detector()
    huge_data = b"\xff\xd8\xff" + b"\x00" * (MAX_FILE_SIZE_BYTES + 4096)
    with pytest.raises(Exception) as exc413:
        detector.validate_image_bytes(huge_data)
    assert getattr(exc413.value, "status_code", None) == 413


# ---------------------------------------------------------------------------
# 5. Taxonomy & Model Information Contract Integrity
# ---------------------------------------------------------------------------


def test_taxonomy_contract_integrity():
    """Verify all 20 classes fulfill required schema constraints."""
    classes_resp = get_classes()
    assert classes_resp.total_classes == 20

    seen_ids = set()
    seen_names = set()
    for c in classes_resp.classes:
        assert c.id not in seen_ids, f"Duplicate ID: {c.id}"
        assert c.name not in seen_names, f"Duplicate class name: {c.name}"
        seen_ids.add(c.id)
        seen_names.add(c.name)

        assert c.mid.startswith("/m/"), f"Invalid MID: {c.mid}"
        assert c.color_hex.startswith("#") and len(c.color_hex) == 7
        assert len(c.color_rgb) == 3
        assert all(0 <= v <= 255 for v in c.color_rgb)


def test_model_info_contract_integrity():
    """Verify model info endpoint provides SLA and benchmark details."""
    model_resp = get_model_info()
    assert model_resp.version == "v1.0.0"
    assert model_resp.num_classes == 21
    assert model_resp.optimal_confidence_threshold == 0.25
    assert model_resp.metrics.mAP_all > 0.20
    assert model_resp.metrics.mAP_novel_classes > 0.10
    assert model_resp.metrics.mAP_coco_shared > 0.30
