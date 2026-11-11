"""
api/tests/test_health.py
Smoke tests for the /healthz endpoint and FastAPI app initialization.
Marked @pytest.mark.stage0.
"""

import pytest

from api.main import app, health


@pytest.mark.stage0
def test_app_metadata() -> None:
    """Verify application metadata is properly configured."""
    assert app.title == "Object Detection API"
    assert app.version == "1.0.0"


@pytest.mark.stage0
def test_healthz_route_registered() -> None:
    """Verify the /healthz route is registered in the routing table."""
    route_paths = list(app.openapi()["paths"].keys())
    assert "/healthz" in route_paths


@pytest.mark.stage0
def test_health_handler_response() -> None:
    """Verify the health handler returns the expected status payload."""
    payload = health()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
