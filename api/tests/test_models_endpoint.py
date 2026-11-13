"""
api/tests/test_models_endpoint.py
---------------------------------
Tests for GET /v1/models endpoint, model registry schema, runtime status, and OpenAPI.
"""

from __future__ import annotations

from api.main import app
from api.routers.model import list_models
from api.schemas import ModelsResponse
from api.services.detector import get_model_cache


def test_list_models_returns_models_response() -> None:
    """Verify list_models() returns valid ModelsResponse with registered models."""
    response = list_models()
    assert isinstance(response, ModelsResponse)
    assert response.total_models >= 2

    model_ids = [m.id for m in response.models]
    assert "fine-tuned" in model_ids
    assert "coco-baseline" in model_ids


def test_list_models_fields_and_status() -> None:
    """Verify all fields, input dimensions, and loaded status for models."""
    cache = get_model_cache()
    # Explicitly load fine-tuned model
    cache.get("fine-tuned")

    response = list_models()
    fine_tuned = next(m for m in response.models if m.id == "fine-tuned")

    assert fine_tuned.display_name
    assert fine_tuned.version.startswith("v")
    assert fine_tuned.input_size == [300, 300]
    assert fine_tuned.num_classes == 20
    assert len(fine_tuned.class_list) == 20
    assert fine_tuned.metrics["mAP_all"] > 0
    assert fine_tuned.loaded is True
    assert fine_tuned.status == "loaded"

    coco_baseline = next(m for m in response.models if m.id == "coco-baseline")
    assert coco_baseline.num_classes == 80
    assert len(coco_baseline.class_list) == 80
    assert isinstance(coco_baseline.loaded, bool)
    assert coco_baseline.status in ("loaded", "ready")


def test_openapi_includes_models_endpoint() -> None:
    """Verify OpenAPI specification registers GET /v1/models."""
    openapi_schema = app.openapi()
    assert "/v1/models" in openapi_schema["paths"]
    assert "get" in openapi_schema["paths"]["/v1/models"]

    endpoint_spec = openapi_schema["paths"]["/v1/models"]["get"]
    assert endpoint_spec["summary"] == "List all registered models with loading status"
    assert "200" in endpoint_spec["responses"]
