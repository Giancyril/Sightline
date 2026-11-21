"""
api/logging_config.py
---------------------
Structured JSON request logging for the Object Detection API.

Each completed request emits a single JSON log line to stdout containing:
  - request_id  : UUID injected by RequestIDMiddleware
  - route       : HTTP method + URL path (e.g. "POST /detect")
  - status      : HTTP status code integer
  - latency_ms  : wall-clock request duration, rounded to 1 decimal place
  - model_id    : value of the ``model_id`` query/body parameter, or null

Privacy contract
----------------
* No image data, pixel values, or base64 strings are ever written to logs.
* No IP addresses, user agents, or personal identifiers are written.
* The ``model_id`` field captures only the model selector string.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.services.metrics import get_metrics_collector

logger = logging.getLogger("api.requests")


def _extract_model_id(request: Request) -> str | None:
    """Return the model_id query parameter if present, else None."""
    return request.query_params.get("model_id") or None


class JSONRequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured JSON log line per completed HTTP request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        request_id = getattr(request.state, "request_id", None)
        model_id = _extract_model_id(request)

        record: dict[str, object] = {
            "request_id": request_id,
            "route": f"{request.method} {request.url.path}",
            "status": response.status_code,
            "latency_ms": latency_ms,
            "model_id": model_id,
        }
        logger.info(json.dumps(record))

        # Update Prometheus metrics
        get_metrics_collector().record_request(
            route=request.url.path,
            method=request.method,
            status=response.status_code,
            latency_s=latency_ms / 1000.0,
        )

        return response


def configure_json_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to emit plain lines (no extra formatting).

    Each log line produced by :class:`JSONRequestLoggingMiddleware` is already
    a valid JSON object, so we use a bare ``%(message)s`` formatter.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Suppress noisy uvicorn access logs (we emit our own structured lines)
    logging.getLogger("uvicorn.access").propagate = False
