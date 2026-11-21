"""
api/routers/metrics.py
----------------------
Endpoint exposing application metrics in Prometheus text exposition format (0.0.4).
Accessible at GET /metrics and GET /v1/metrics for Prometheus scrapers and ops dashboards.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.services.metrics import get_metrics_collector
from ml.eval.drift import get_drift_detector

router = APIRouter(tags=["Metrics"])


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    description="Returns API request counts, error counts, latency histograms, and inference metrics in Prometheus text format.",
)
@router.get(
    "/v1/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
)
def prometheus_metrics() -> PlainTextResponse:
    """Render metrics in standard Prometheus text format."""
    text = get_metrics_collector().render_prometheus_text()
    return PlainTextResponse(
        content=text,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get(
    "/v1/drift",
    summary="Confidence distribution drift status",
    description="Returns Population Stability Index (PSI), baseline vs live distributions, and nominal/warning status.",
)
def get_drift_status() -> dict[str, Any]:
    """Evaluate real-time confidence distribution drift against validation baseline."""
    return get_drift_detector().check_drift().to_dict()


@router.get(
    "/v1/operations",
    summary="Operations dashboard telemetry",
    description="Returns live p50/p95 latency, error rates, confidence distribution drift, and model status.",
)
def get_operations_telemetry() -> dict[str, Any]:
    """Consolidated telemetry payload for the frontend Operations dashboard."""
    ops_summary = get_metrics_collector().get_operations_summary()
    drift_report = get_drift_detector().check_drift().to_dict()

    return {
        "model_id": "fine-tuned",
        "model_version": "v1.1.0",
        "display_name": "SSD MobileNet V2 (Open Images 20-Class)",
        **ops_summary,
        "drift": drift_report,
    }
