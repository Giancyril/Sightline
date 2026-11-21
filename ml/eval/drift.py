"""
ml/eval/drift.py
----------------
Confidence distribution drift detection against the validation baseline.

Implements:
  - Population Stability Index (PSI) calculation between expected and actual distributions.
  - Thread-safe rolling window of live confidence scores from inference.
  - Baseline distribution derived from validation set calibration fixtures.
  - Automated status evaluation ("ok" vs "warning") based on configurable PSI threshold.
"""

from __future__ import annotations

import collections
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Sequence

# Validation baseline counts across 10 confidence bins [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
# Sourced from models/fine-tuned/calibration.json (2,000 validation detections)
VALIDATION_BASELINE_COUNTS: tuple[int, ...] = (
    55,
    82,
    130,
    165,
    195,
    225,
    250,
    280,
    325,
    293,
)

DEFAULT_PSI_WARNING_THRESHOLD: float = 0.20
DEFAULT_MIN_SAMPLES: int = 30
DEFAULT_WINDOW_SIZE: int = 500


def calculate_psi(
    expected: Sequence[float],
    actual: Sequence[float],
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate the Population Stability Index (PSI) between two distributions.

    PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))

    Args:
        expected: Baseline/expected probabilities or counts per bin.
        actual: Live/observed probabilities or counts per bin.
        epsilon: Small smoothing constant to prevent division by zero or log(0).

    Returns:
        PSI float value (>= 0.0). Higher values indicate greater distribution shift.
    """
    if len(expected) != len(actual):
        raise ValueError(
            f"Bins count mismatch: expected {len(expected)} bins, got {len(actual)}"
        )
    if not expected or not actual:
        return 0.0

    exp_sum = float(sum(expected))
    act_sum = float(sum(actual))

    if exp_sum <= 0.0 or act_sum <= 0.0:
        return 0.0

    # Normalize to proportions
    exp_props = [e / exp_sum for e in expected]
    act_props = [a / act_sum for a in actual]

    psi = 0.0
    for e_p, a_p in zip(exp_props, act_props, strict=False):
        # Apply epsilon smoothing
        e_smooth = max(e_p, epsilon)
        a_smooth = max(a_p, epsilon)
        psi += (a_smooth - e_smooth) * math.log(a_smooth / e_smooth)

    return round(max(0.0, psi), 4)


class RollingConfidenceBuffer:
    """Thread-safe FIFO rolling buffer storing recent confidence scores."""

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._buffer: collections.deque[float] = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, score: float) -> None:
        """Add a single confidence score clamped to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(score)))
        with self._lock:
            self._buffer.append(clamped)

    def extend(self, scores: Sequence[float]) -> None:
        """Add multiple confidence scores."""
        with self._lock:
            for s in scores:
                self._buffer.append(max(0.0, min(1.0, float(s))))

    def clear(self) -> None:
        """Clear all stored confidence scores."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_scores(self) -> list[float]:
        """Return a snapshot copy of current scores."""
        with self._lock:
            return list(self._buffer)

    def compute_binned_distribution(
        self, num_bins: int = 10
    ) -> tuple[list[float], list[float], list[int]]:
        """
        Bin current buffer contents into uniform intervals in [0.0, 1.0].

        Returns:
            (bin_edges, proportions, raw_counts)
        """
        step = 1.0 / num_bins
        edges = [round(i * step, 2) for i in range(num_bins + 1)]
        counts = [0] * num_bins

        with self._lock:
            scores_snapshot = list(self._buffer)

        for score in scores_snapshot:
            # Index for score in [0.0, 1.0]; clamp upper edge 1.0 to last bin
            idx = int(score / step)
            if idx >= num_bins:
                idx = num_bins - 1
            counts[idx] += 1

        total = len(scores_snapshot)
        proportions = (
            [round(c / total, 4) for c in counts] if total > 0 else [0.0] * num_bins
        )
        return edges, proportions, counts


@dataclass
class DriftReport:
    """Drift check evaluation outcome."""

    status: str  # "ok" or "warning"
    psi: float
    threshold: float
    sample_count: int
    min_samples: int
    min_samples_reached: bool
    baseline_distribution: list[float]
    live_distribution: list[float]
    bin_edges: list[float]
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DriftDetector:
    """
    Monitors inference confidence scores against a baseline validation distribution.
    Computes PSI on a sliding window and triggers 'warning' when shift exceeds threshold.
    """

    _instance: ClassVar[DriftDetector | None] = None
    _singleton_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        baseline: Sequence[float] | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        psi_threshold: float = DEFAULT_PSI_WARNING_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples
        self.buffer = RollingConfidenceBuffer(window_size=window_size)

        # Baseline proportions (10 bins)
        raw_baseline = (
            list(baseline) if baseline is not None else list(VALIDATION_BASELINE_COUNTS)
        )
        base_sum = float(sum(raw_baseline))
        self.baseline_proportions: list[float] = [
            round(c / base_sum, 4) for c in raw_baseline
        ]
        self.num_bins = len(self.baseline_proportions)

    @classmethod
    def get_instance(cls) -> DriftDetector:
        """Global singleton instance accessor."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        with cls._singleton_lock:
            cls._instance = None

    def record_confidence(self, score: float) -> None:
        """Ingest a single live inference confidence score."""
        self.buffer.add(score)

    def record_confidences(self, scores: Sequence[float]) -> None:
        """Ingest multiple live inference confidence scores."""
        self.buffer.extend(scores)

    def reset(self) -> None:
        """Clear rolling buffer."""
        self.buffer.clear()

    def check_drift(self) -> DriftReport:
        """
        Evaluate distribution shift against baseline.

        Returns:
            DriftReport with status 'ok' or 'warning', PSI value, and distributions.
        """
        edges, live_props, live_counts = self.buffer.compute_binned_distribution(
            num_bins=self.num_bins
        )
        n_samples = len(self.buffer)

        # Insufficient data condition
        if n_samples < self.min_samples:
            return DriftReport(
                status="ok",
                psi=0.0,
                threshold=self.psi_threshold,
                sample_count=n_samples,
                min_samples=self.min_samples,
                min_samples_reached=False,
                baseline_distribution=self.baseline_proportions,
                live_distribution=live_props,
                bin_edges=edges,
                message=f"Insufficient samples ({n_samples}/{self.min_samples}) for drift evaluation; status is nominal.",
            )

        psi = calculate_psi(self.baseline_proportions, live_props)
        is_drifted = psi >= self.psi_threshold
        status = "warning" if is_drifted else "ok"

        if is_drifted:
            msg = (
                f"Confidence drift detected: PSI {psi:.4f} exceeds threshold {self.psi_threshold:.2f} "
                f"across {n_samples} live observations."
            )
        else:
            msg = (
                f"Confidence distribution stable: PSI {psi:.4f} within threshold {self.psi_threshold:.2f} "
                f"across {n_samples} live observations."
            )

        return DriftReport(
            status=status,
            psi=psi,
            threshold=self.psi_threshold,
            sample_count=n_samples,
            min_samples=self.min_samples,
            min_samples_reached=True,
            baseline_distribution=self.baseline_proportions,
            live_distribution=live_props,
            bin_edges=edges,
            message=msg,
        )

    @classmethod
    def from_calibration_file(
        cls,
        calibration_path: str | Path,
        window_size: int = DEFAULT_WINDOW_SIZE,
        psi_threshold: float = DEFAULT_PSI_WARNING_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> DriftDetector:
        """Initialize DriftDetector using bins from a calibration.json fixture."""
        p = Path(calibration_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        bins = data.get("bins_after") or data.get("bins_before") or []
        counts = [b["count"] for b in bins]
        if not counts:
            raise ValueError(f"No calibration bin counts found in {calibration_path}")
        return cls(
            baseline=counts,
            window_size=window_size,
            psi_threshold=psi_threshold,
            min_samples=min_samples,
        )


def get_drift_detector() -> DriftDetector:
    """Return the global DriftDetector singleton."""
    return DriftDetector.get_instance()
