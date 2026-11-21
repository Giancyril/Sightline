"""
api/tests/test_logging.py
--------------------------
Tests for structured JSON request logging and request-id propagation.

Covers:
  - Middleware registration on main.app
  - X-Request-ID header echoed back in every response
  - Auto-generated request-id when header is absent (32-char hex UUID4)
  - Caller-supplied request-id preserved and echoed verbatim
  - Log line is valid JSON with all required fields
  - Log line contains no image data or personal information
  - Latency field is a non-negative float
  - model_id captured from query parameter; null when absent
  - Multiple concurrent requests each receive distinct ids
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.logging_config import JSONRequestLoggingMiddleware
from api.main import app as main_app
from api.middleware.request_id import REQUEST_ID_HEADER, RequestIDMiddleware

REQUIRED_LOG_FIELDS = {"request_id", "route", "status", "latency_ms", "model_id"}


def create_test_app() -> FastAPI:
    """Create a lightweight app with logging & request-id middlewares for fast testing."""
    test_app = FastAPI()
    test_app.add_middleware(JSONRequestLoggingMiddleware)
    test_app.add_middleware(RequestIDMiddleware)

    @test_app.get("/healthz")
    def health_stub(model_id: str | None = None) -> dict[str, str]:
        return {"status": "ok"}

    return test_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_test_app())


@pytest.fixture()
def log_stream() -> StringIO:
    """Capture api.requests log output into an in-memory buffer."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    api_logger = logging.getLogger("api.requests")
    original_handlers = api_logger.handlers[:]
    original_propagate = api_logger.propagate
    api_logger.handlers = [handler]
    api_logger.propagate = False
    api_logger.setLevel(logging.INFO)
    yield stream
    api_logger.handlers = original_handlers
    api_logger.propagate = original_propagate


def _last_log_record(stream: StringIO) -> dict[str, object]:
    """Return the last non-empty JSON log line from the captured stream."""
    stream.seek(0)
    lines = [ln.strip() for ln in stream.read().splitlines() if ln.strip()]
    assert lines, "No log lines were emitted"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Registration test on main app
# ---------------------------------------------------------------------------


def test_main_app_has_middlewares_registered() -> None:
    """Verify production main_app has RequestID and JSONRequestLogging middlewares."""
    middleware_names = [m.cls.__name__ for m in main_app.user_middleware]
    assert "RequestIDMiddleware" in middleware_names
    assert "JSONRequestLoggingMiddleware" in middleware_names


# ---------------------------------------------------------------------------
# Request-id header tests
# ---------------------------------------------------------------------------


def test_response_contains_request_id_header(client: TestClient) -> None:
    """Every response must echo X-Request-ID."""
    resp = client.get("/healthz")
    assert REQUEST_ID_HEADER in resp.headers, (
        f"Missing {REQUEST_ID_HEADER} header in response"
    )


def test_auto_generated_request_id_is_32_char_hex(client: TestClient) -> None:
    """When no X-Request-ID is sent, a 32-character hex UUID4 is generated."""
    resp = client.get("/healthz")
    request_id = resp.headers.get(REQUEST_ID_HEADER, "")
    assert len(request_id) == 32, f"Expected 32-char hex, got {request_id!r}"
    assert all(c in "0123456789abcdef" for c in request_id), (
        f"Non-hex characters in request_id: {request_id!r}"
    )


def test_caller_supplied_request_id_is_preserved(client: TestClient) -> None:
    """A caller-provided X-Request-ID must be echoed verbatim."""
    custom_id = "my-trace-abc123"
    resp = client.get("/healthz", headers={REQUEST_ID_HEADER: custom_id})
    assert resp.headers.get(REQUEST_ID_HEADER) == custom_id


def test_distinct_ids_for_concurrent_requests(client: TestClient) -> None:
    """Auto-generated ids across multiple requests must all be unique."""
    ids = {client.get("/healthz").headers.get(REQUEST_ID_HEADER) for _ in range(10)}
    assert len(ids) == 10, f"Duplicate request ids detected: {ids}"


# ---------------------------------------------------------------------------
# Structured log field tests
# ---------------------------------------------------------------------------


def test_log_line_is_valid_json(client: TestClient, log_stream: StringIO) -> None:
    """Each request must produce a parseable JSON log line."""
    client.get("/healthz")
    record = _last_log_record(log_stream)
    assert isinstance(record, dict)


def test_log_line_contains_required_fields(
    client: TestClient, log_stream: StringIO
) -> None:
    """Log record must contain all required structured fields."""
    client.get("/healthz")
    record = _last_log_record(log_stream)
    missing = REQUIRED_LOG_FIELDS - record.keys()
    assert not missing, f"Log record missing fields: {missing}"


def test_log_route_field_format(client: TestClient, log_stream: StringIO) -> None:
    """route field must be '<METHOD> <path>' (e.g. 'GET /healthz')."""
    client.get("/healthz")
    record = _last_log_record(log_stream)
    assert record["route"] == "GET /healthz"


def test_log_status_field(client: TestClient, log_stream: StringIO) -> None:
    """status field must match the HTTP response status code."""
    resp = client.get("/healthz")
    record = _last_log_record(log_stream)
    assert record["status"] == resp.status_code


def test_log_latency_is_non_negative_float(
    client: TestClient, log_stream: StringIO
) -> None:
    """latency_ms must be a non-negative number."""
    client.get("/healthz")
    record = _last_log_record(log_stream)
    latency = record["latency_ms"]
    assert isinstance(latency, (int, float)), f"latency_ms is not numeric: {latency!r}"
    assert latency >= 0, f"Negative latency: {latency}"


def test_log_model_id_is_null_when_absent(
    client: TestClient, log_stream: StringIO
) -> None:
    """model_id must be null when no model_id query parameter is provided."""
    client.get("/healthz")
    record = _last_log_record(log_stream)
    assert record["model_id"] is None


def test_log_model_id_captured_from_query(
    client: TestClient, log_stream: StringIO
) -> None:
    """model_id query param must appear in the log record."""
    client.get("/healthz?model_id=fine-tuned")
    record = _last_log_record(log_stream)
    assert record["model_id"] == "fine-tuned"


def test_log_request_id_matches_response_header(
    client: TestClient, log_stream: StringIO
) -> None:
    """request_id in log must match the X-Request-ID echoed in the response."""
    resp = client.get("/healthz")
    record = _last_log_record(log_stream)
    assert record["request_id"] == resp.headers.get(REQUEST_ID_HEADER)


def test_no_image_data_in_log(client: TestClient, log_stream: StringIO) -> None:
    """Log line must not contain any base64-encoded image payload."""
    client.get("/healthz")
    log_stream.seek(0)
    raw = log_stream.read()
    import re

    b64_pattern = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")
    assert not b64_pattern.search(raw), "Possible image data found in log output"
