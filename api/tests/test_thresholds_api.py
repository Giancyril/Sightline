"""
api/tests/test_thresholds_api.py
--------------------------------
Integration and unit tests for Day 3 Commit 7:
  - GET /v1/models/{id}/thresholds (per-class table, missing-file fallback, 404)
  - POST /v1/detect with threshold_mode ('global', 'recommended', 'custom')
  - Score calibration flag and mode reporting in DetectionResponse
  - POST /v1/compare with threshold_mode
  - Input validation error handling (invalid mode, invalid custom thresholds)
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi import HTTPException, Response, UploadFile
from PIL import Image

from api.routers.compare import compare_models
from api.routers.detect import detect_objects
from api.routers.model import get_model_thresholds
from api.schemas import CompareResponse, DetectionResponse, ModelThresholdsResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_image(
    width: int = 100,
    height: int = 100,
    color: tuple[int, int, int] = (120, 140, 160),
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_upload_file(data: bytes, filename: str = "test.jpg") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data))


# ---------------------------------------------------------------------------
# 1. GET /v1/models/{id}/thresholds
# ---------------------------------------------------------------------------


def test_get_fine_tuned_thresholds_success() -> None:
    """GET /v1/models/fine-tuned/thresholds returns 200 with per-class table."""
    res = get_model_thresholds("fine-tuned")
    assert isinstance(res, ModelThresholdsResponse)
    assert res.model_id == "fine-tuned"
    assert res.rule == "best_f1"
    assert res.global_threshold == pytest.approx(0.25)
    assert len(res.thresholds) == 20
    assert len(res.per_class) == 20
    assert "Person" in res.thresholds
    assert res.thresholds["Person"] == pytest.approx(0.26)
    assert len(res.fallbacks) > 0


def test_get_coco_baseline_thresholds_raises_404() -> None:
    """GET /v1/models/coco-baseline/thresholds raises 404 because no threshold file exists."""
    with pytest.raises(HTTPException) as exc_info:
        get_model_thresholds("coco-baseline")
    assert exc_info.value.status_code == 404
    assert "does not have recommended thresholds" in str(exc_info.value.detail)


def test_get_unknown_model_thresholds_raises_404() -> None:
    """GET /v1/models/unknown-xyz/thresholds raises 404 for unknown model."""
    with pytest.raises(HTTPException) as exc_info:
        get_model_thresholds("unknown-xyz")
    assert exc_info.value.status_code == 404
    assert "Unknown model_id" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# 2. POST /v1/detect with threshold_mode
# ---------------------------------------------------------------------------


def test_detect_default_mode_is_global() -> None:
    """POST /v1/detect without threshold_mode defaults to 'global' and calibration=False."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    res = detect_objects(response=resp, file=upload, model_id="fine-tuned")
    assert isinstance(res, DetectionResponse)
    assert res.threshold_mode == "global"
    assert res.calibration_applied is False
    assert res.thresholds_applied.threshold_mode == "global"
    assert res.thresholds_applied.calibration_applied is False


def test_detect_recommended_mode_fine_tuned() -> None:
    """POST /v1/detect with threshold_mode='recommended' applies calibration and per-class thresholds."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    res = detect_objects(
        response=resp,
        file=upload,
        model_id="fine-tuned",
        threshold_mode="recommended",
    )
    assert isinstance(res, DetectionResponse)
    assert res.threshold_mode == "recommended"
    assert res.calibration_applied is True
    assert res.thresholds_applied.threshold_mode == "recommended"
    assert res.thresholds_applied.calibration_applied is True


def test_detect_recommended_mode_fallback_for_model_without_files() -> None:
    """POST /v1/detect with threshold_mode='recommended' for coco-baseline falls back gracefully."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    res = detect_objects(
        response=resp,
        file=upload,
        model_id="coco-baseline",
        threshold_mode="recommended",
    )
    assert isinstance(res, DetectionResponse)
    # Model has no thresholds file, so calibration is not applied
    assert res.calibration_applied is False


def test_detect_custom_mode_valid() -> None:
    """POST /v1/detect with threshold_mode='custom' and valid JSON thresholds."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    custom_map = json.dumps({"Person": 0.50, "Car": 0.40})
    res = detect_objects(
        response=resp,
        file=upload,
        model_id="fine-tuned",
        threshold_mode="custom",
        custom_thresholds=custom_map,
    )
    assert isinstance(res, DetectionResponse)
    assert res.threshold_mode == "custom"
    assert res.calibration_applied is False


def test_detect_custom_mode_missing_thresholds_raises_400() -> None:
    """POST /v1/detect with threshold_mode='custom' but missing custom_thresholds raises 400."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    with pytest.raises(HTTPException) as exc_info:
        detect_objects(
            response=resp,
            file=upload,
            model_id="fine-tuned",
            threshold_mode="custom",
            custom_thresholds=None,
        )
    assert exc_info.value.status_code == 400
    assert "custom_thresholds JSON string is required" in str(exc_info.value.detail)


def test_detect_custom_mode_invalid_json_raises_400() -> None:
    """POST /v1/detect with invalid custom_thresholds JSON raises 400."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    with pytest.raises(HTTPException) as exc_info:
        detect_objects(
            response=resp,
            file=upload,
            model_id="fine-tuned",
            threshold_mode="custom",
            custom_thresholds="{bad json",
        )
    assert exc_info.value.status_code == 400
    assert "Invalid custom_thresholds" in str(exc_info.value.detail)


def test_detect_custom_mode_out_of_range_threshold_raises_400() -> None:
    """POST /v1/detect with threshold value outside [0, 1] raises 400."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    with pytest.raises(HTTPException) as exc_info:
        detect_objects(
            response=resp,
            file=upload,
            model_id="fine-tuned",
            threshold_mode="custom",
            custom_thresholds=json.dumps({"Person": 1.5}),
        )
    assert exc_info.value.status_code == 400
    assert "between 0.0 and 1.0" in str(exc_info.value.detail)


def test_detect_invalid_threshold_mode_raises_400() -> None:
    """POST /v1/detect with unsupported threshold_mode raises 400."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)
    resp = Response()

    with pytest.raises(HTTPException) as exc_info:
        detect_objects(
            response=resp,
            file=upload,
            model_id="fine-tuned",
            threshold_mode="super_mode",
        )
    assert exc_info.value.status_code == 400
    assert "Invalid threshold_mode" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# 3. POST /v1/compare with threshold_mode
# ---------------------------------------------------------------------------


def test_compare_with_recommended_threshold_mode() -> None:
    """POST /v1/compare accepts threshold_mode='recommended'."""
    img_bytes = _make_dummy_image()
    upload = _make_upload_file(img_bytes)

    res = compare_models(
        file=upload,
        model_a="fine-tuned",
        model_b="coco-baseline",
        threshold_mode="recommended",
    )
    assert isinstance(res, CompareResponse)
    assert res.model_a_result.threshold_mode == "recommended"
    assert res.model_a_result.calibration_applied is True
