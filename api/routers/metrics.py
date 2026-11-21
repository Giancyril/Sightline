"""
api/routers/metrics.py
----------------------
Endpoint exposing application metrics in Prometheus text exposition format (0.0.4).
Accessible at GET /metrics and GET /v1/metrics for Prometheus scrapers and ops dashboards.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.services.metrics import get_metrics_collector

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
