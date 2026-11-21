"""
api/services/metrics.py
-----------------------
Thread-safe Prometheus metrics collector for the Object Detection API.

Collects and renders metrics in Prometheus text format (version 0.0.4):
  - detection_api_requests_total          : Counter (route, method, status)
  - detection_api_errors_total            : Counter (route, status)
  - detection_api_request_latency_seconds : Histogram (le buckets, sum, count)
  - detection_api_detections_per_image    : Histogram (box count per image)
  - detection_api_confidence_scores       : Histogram (confidence score distribution)
  - detection_api_model_usage_total       : Counter (model_id invocations)
"""

from __future__ import annotations

import threading
from typing import ClassVar

LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

DETECTION_COUNT_BUCKETS: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
)

CONFIDENCE_BUCKETS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)


class MetricsCollector:
    """Thread-safe collector for API request and inference metrics."""

    _instance: ClassVar[MetricsCollector | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self.reset()

    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Singleton accessor."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def reset(self) -> None:
        """Reset all metrics to initial state (useful in tests)."""
        with self._mutex:
            # (route, method, status) -> count
            self.requests_total: dict[tuple[str, str, str], float] = {}
            # (route, status) -> count
            self.errors_total: dict[tuple[str, str], float] = {}
            # model_id -> count
            self.model_usage_total: dict[str, float] = {}

            # Histograms: {bucket_le: count}, sum, count
            self.latency_buckets: dict[float, float] = {b: 0.0 for b in LATENCY_BUCKETS}
            self.latency_sum: float = 0.0
            self.latency_count: float = 0.0

            self.detections_buckets: dict[float, float] = {
                b: 0.0 for b in DETECTION_COUNT_BUCKETS
            }
            self.detections_sum: float = 0.0
            self.detections_count: float = 0.0

            self.confidence_buckets: dict[float, float] = {
                b: 0.0 for b in CONFIDENCE_BUCKETS
            }
            self.confidence_sum: float = 0.0
            self.confidence_count: float = 0.0

    def record_request(
        self, route: str, method: str, status: int, latency_s: float
    ) -> None:
        """Record an inbound HTTP request and its latency."""
        route_key = route or "/unknown"
        method_key = method.upper()
        status_key = str(status)

        with self._mutex:
            # Request count
            key = (route_key, method_key, status_key)
            self.requests_total[key] = self.requests_total.get(key, 0.0) + 1.0

            # Error count if 4xx or 5xx
            if status >= 400:
                err_key = (route_key, status_key)
                self.errors_total[err_key] = self.errors_total.get(err_key, 0.0) + 1.0

            # Latency histogram
            self.latency_sum += latency_s
            self.latency_count += 1.0
            for b in LATENCY_BUCKETS:
                if latency_s <= b:
                    self.latency_buckets[b] += 1.0

    def record_detections(
        self,
        model_id: str,
        detection_count: int,
        confidences: list[float] | tuple[float, ...],
    ) -> None:
        """Record inference detection counts and confidence score distribution."""
        with self._mutex:
            # Model usage
            mid = model_id or "unknown"
            self.model_usage_total[mid] = self.model_usage_total.get(mid, 0.0) + 1.0

            # Detections per image
            cnt = float(detection_count)
            self.detections_sum += cnt
            self.detections_count += 1.0
            for b in DETECTION_COUNT_BUCKETS:
                if cnt <= b:
                    self.detections_buckets[b] += 1.0

            # Confidence scores
            for conf in confidences:
                c = float(conf)
                self.confidence_sum += c
                self.confidence_count += 1.0
                for b in CONFIDENCE_BUCKETS:
                    if c <= b:
                        self.confidence_buckets[b] += 1.0

    def render_prometheus_text(self) -> str:
        """Generate Prometheus exposition text format (0.0.4)."""
        lines: list[str] = []

        with self._mutex:
            # 1. requests_total
            lines.append(
                "# HELP detection_api_requests_total Total number of HTTP requests processed."
            )
            lines.append("# TYPE detection_api_requests_total counter")
            if self.requests_total:
                for (r, m, s), val in sorted(self.requests_total.items()):
                    lines.append(
                        f'detection_api_requests_total{{method="{m}",route="{r}",status="{s}"}} {val}'
                    )
            else:
                lines.append(
                    'detection_api_requests_total{method="GET",route="/healthz",status="200"} 0.0'
                )

            # 2. errors_total
            lines.append(
                "# HELP detection_api_errors_total Total number of HTTP error responses (status >= 400)."
            )
            lines.append("# TYPE detection_api_errors_total counter")
            if self.errors_total:
                for (r, s), val in sorted(self.errors_total.items()):
                    lines.append(
                        f'detection_api_errors_total{{route="{r}",status="{s}"}} {val}'
                    )
            else:
                lines.append(
                    'detection_api_errors_total{route="/healthz",status="500"} 0.0'
                )

            # 3. request_latency_seconds
            lines.append(
                "# HELP detection_api_request_latency_seconds Request latency in seconds."
            )
            lines.append("# TYPE detection_api_request_latency_seconds histogram")
            for b in LATENCY_BUCKETS:
                lines.append(
                    f'detection_api_request_latency_seconds_bucket{{le="{b}"}} {self.latency_buckets[b]}'
                )
            lines.append(
                f'detection_api_request_latency_seconds_bucket{{le="+Inf"}} {self.latency_count}'
            )
            lines.append(
                f"detection_api_request_latency_seconds_sum {round(self.latency_sum, 6)}"
            )
            lines.append(
                f"detection_api_request_latency_seconds_count {self.latency_count}"
            )

            # 4. detections_per_image
            lines.append(
                "# HELP detection_api_detections_per_image Number of detected bounding boxes per image."
            )
            lines.append("# TYPE detection_api_detections_per_image histogram")
            for b in DETECTION_COUNT_BUCKETS:
                val = int(b) if b.is_integer() else b
                lines.append(
                    f'detection_api_detections_per_image_bucket{{le="{val}"}} {self.detections_buckets[b]}'
                )
            lines.append(
                f'detection_api_detections_per_image_bucket{{le="+Inf"}} {self.detections_count}'
            )
            lines.append(
                f"detection_api_detections_per_image_sum {round(self.detections_sum, 4)}"
            )
            lines.append(
                f"detection_api_detections_per_image_count {self.detections_count}"
            )

            # 5. confidence_scores
            lines.append(
                "# HELP detection_api_confidence_scores Distribution of detection confidence scores."
            )
            lines.append("# TYPE detection_api_confidence_scores histogram")
            for b in CONFIDENCE_BUCKETS:
                lines.append(
                    f'detection_api_confidence_scores_bucket{{le="{b}"}} {self.confidence_buckets[b]}'
                )
            lines.append(
                f'detection_api_confidence_scores_bucket{{le="+Inf"}} {self.confidence_count}'
            )
            lines.append(
                f"detection_api_confidence_scores_sum {round(self.confidence_sum, 4)}"
            )
            lines.append(
                f"detection_api_confidence_scores_count {self.confidence_count}"
            )

            # 6. model_usage_total
            lines.append(
                "# HELP detection_api_model_usage_total Total inferences dispatched per model."
            )
            lines.append("# TYPE detection_api_model_usage_total counter")
            if self.model_usage_total:
                for mid, val in sorted(self.model_usage_total.items()):
                    lines.append(
                        f'detection_api_model_usage_total{{model_id="{mid}"}} {val}'
                    )
            else:
                lines.append(
                    'detection_api_model_usage_total{model_id="fine-tuned"} 0.0'
                )

        return "\n".join(lines) + "\n"


def get_metrics_collector() -> MetricsCollector:
    """Return the global MetricsCollector instance."""
    return MetricsCollector.get_instance()
