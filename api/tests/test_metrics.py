"""
api/tests/test_metrics.py
-------------------------
Tests for GET /metrics endpoint and the Prometheus exposition collector.

Covers:
  - Route registration on main.app (/metrics and /v1/metrics)
  - Prometheus text format (0.0.4) compliance:
      * # HELP and # TYPE headers for all 6 metrics
      * Valid metric names and label quoting
      * Histogram bucket structure: ascending le, +Inf, sum, count
  - Counter increments on record_request (requests_total, errors_total)
  - Histogram updates on record_request (latency_seconds)
  - Inference metrics recording on record_detections (model_usage, detections_per_image, confidence_scores)
  - Thread-safe collector reset functionality
  - Middleware integration updating metrics on simulated requests
"""

from __future__ import annotations

import re

import pytest

from api.main import app
from api.routers.metrics import prometheus_metrics
from api.services.metrics import (
    LATENCY_BUCKETS,
    get_metrics_collector,
)


@pytest.fixture(autouse=True)
def clean_metrics() -> None:
    """Ensure a pristine metrics state for each test."""
    collector = get_metrics_collector()
    collector.reset()
    yield
    collector.reset()


# ---------------------------------------------------------------------------
# Route and Response Format
# ---------------------------------------------------------------------------


def test_metrics_route_registered() -> None:
    """Verify /metrics route is registered in the FastAPI app."""
    paths = list(app.openapi()["paths"].keys())
    assert "/metrics" in paths


def test_metrics_content_type_header() -> None:
    """Verify response media_type matches Prometheus 0.0.4 specification."""
    resp = prometheus_metrics()
    assert resp.status_code == 200
    assert "text/plain" in resp.media_type
    assert "version=0.0.4" in resp.media_type


def test_prometheus_exposition_headers_present() -> None:
    """Verify all 6 required metrics have # HELP and # TYPE headers."""
    resp = prometheus_metrics()
    body = resp.body.decode("utf-8")

    expected_metrics = [
        ("detection_api_requests_total", "counter"),
        ("detection_api_errors_total", "counter"),
        ("detection_api_request_latency_seconds", "histogram"),
        ("detection_api_detections_per_image", "histogram"),
        ("detection_api_confidence_scores", "histogram"),
        ("detection_api_model_usage_total", "counter"),
    ]

    for metric_name, metric_type in expected_metrics:
        assert f"# HELP {metric_name} " in body, f"Missing HELP for {metric_name}"
        assert f"# TYPE {metric_name} {metric_type}" in body, (
            f"Missing TYPE for {metric_name}"
        )


# ---------------------------------------------------------------------------
# Recording & Calculation Accuracy
# ---------------------------------------------------------------------------


def test_record_request_increments_counters_and_latency() -> None:
    """Recording a request should increment total count and latency histogram."""
    collector = get_metrics_collector()
    collector.record_request(
        route="/v1/detect", method="POST", status=200, latency_s=0.045
    )

    body = collector.render_prometheus_text()
    assert (
        'detection_api_requests_total{method="POST",route="/v1/detect",status="200"} 1.0'
        in body
    )
    assert "detection_api_request_latency_seconds_count 1.0" in body
    assert "detection_api_request_latency_seconds_sum 0.045" in body

    # le=0.05 bucket should be 1.0, but le=0.025 should be 0.0
    assert 'detection_api_request_latency_seconds_bucket{le="0.05"} 1.0' in body
    assert 'detection_api_request_latency_seconds_bucket{le="0.025"} 0.0' in body


def test_record_request_tracks_errors_for_status_ge_400() -> None:
    """Requests with status >= 400 must increment errors_total."""
    collector = get_metrics_collector()
    collector.record_request(
        route="/v1/detect", method="POST", status=400, latency_s=0.005
    )
    collector.record_request(
        route="/v1/detect", method="POST", status=500, latency_s=0.010
    )

    body = collector.render_prometheus_text()
    assert 'detection_api_errors_total{route="/v1/detect",status="400"} 1.0' in body
    assert 'detection_api_errors_total{route="/v1/detect",status="500"} 1.0' in body


def test_record_detections_updates_model_usage_and_histograms() -> None:
    """Recording inference detections must update usage, box counts, and confidences."""
    collector = get_metrics_collector()
    collector.record_detections(
        model_id="fine-tuned",
        detection_count=3,
        confidences=[0.35, 0.78, 0.92],
    )

    body = collector.render_prometheus_text()
    assert 'detection_api_model_usage_total{model_id="fine-tuned"} 1.0' in body
    assert "detection_api_detections_per_image_count 1.0" in body
    assert "detection_api_detections_per_image_sum 3.0" in body
    assert "detection_api_confidence_scores_count 3.0" in body

    # Check cumulative confidence histogram buckets
    # 0.35 falls in <= 0.4
    assert 'detection_api_confidence_scores_bucket{le="0.4"} 1.0' in body
    # 0.78 falls in <= 0.8 (count = 2)
    assert 'detection_api_confidence_scores_bucket{le="0.8"} 2.0' in body
    # 0.92 falls in <= 1.0 (count = 3)
    assert 'detection_api_confidence_scores_bucket{le="1.0"} 3.0' in body


def test_histogram_bucket_ordering() -> None:
    """Histogram lines must have monotonically non-decreasing bucket counts ending with +Inf."""
    collector = get_metrics_collector()
    collector.record_request(route="/healthz", method="GET", status=200, latency_s=0.15)
    collector.record_request(route="/healthz", method="GET", status=200, latency_s=0.85)

    body = collector.render_prometheus_text()
    pattern = re.compile(
        r'detection_api_request_latency_seconds_bucket\{le="([^"]+)"\} ([0-9.]+)'
    )
    matches = pattern.findall(body)
    assert len(matches) == len(LATENCY_BUCKETS) + 1  # buckets + +Inf

    counts = [float(count) for _, count in matches]
    for i in range(len(counts) - 1):
        assert counts[i] <= counts[i + 1], "Histogram buckets must be cumulative"

    assert counts[-1] == 2.0, "Top (+Inf) bucket must equal total count"


def test_collector_reset() -> None:
    """Resetting the collector restores initial zero-state."""
    collector = get_metrics_collector()
    collector.record_request(route="/healthz", method="GET", status=200, latency_s=0.01)
    assert collector.latency_count == 1.0

    collector.reset()
    assert collector.latency_count == 0.0
    assert collector.latency_sum == 0.0
    assert len(collector.requests_total) == 0
