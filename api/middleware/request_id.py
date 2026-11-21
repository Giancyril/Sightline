"""
api/middleware/request_id.py
----------------------------
ASGI middleware that attaches a unique request-id to every inbound request.

Behaviour
---------
* Reads ``X-Request-ID`` from the incoming headers; if absent, generates a
  compact UUID4 (hex, no hyphens).
* Stores the id in ``request.state.request_id`` for downstream handlers.
* Echoes the id back via the ``X-Request-ID`` response header so callers
  can correlate logs with their own tooling.
* Never logs, stores, or forwards image data or any personal information.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a UUID request-id into each request's state and response headers."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
