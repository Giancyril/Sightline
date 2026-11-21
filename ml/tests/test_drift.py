"""
ml/tests/test_drift.py
----------------------
Unit and fixture tests for confidence drift detection and Population Stability Index (PSI).

Covers:
  - calculate_psi mathematical correctness, edge cases, and symmetry properties
  - RollingConfidenceBuffer FIFO eviction, clamping, and binned distributions
  - DriftDetector status ("ok" vs "warning") on synthetic and fixture distributions
  - Integration with models/fine-tuned/calibration.json validation fixture
  - Insufficient sample safeguards and singleton lifecycle
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.eval.drift import (
    DEFAULT_PSI_WARNING_THRESHOLD,
    VALIDATION_BASELINE_COUNTS,
    DriftDetector,
    RollingConfidenceBuffer,
    calculate_psi,
    get_drift_detector,
)


@pytest.fixture(autouse=True)
def clean_detector() -> None:
    """Ensure a clean drift detector singleton between tests."""
    detector = get_drift_detector()
    detector.reset()
    yield
    detector.reset()


# ---------------------------------------------------------------------------
# 1. calculate_psi Unit Tests
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_is_zero() -> None:
    """Two identical distributions must have PSI = 0.0."""
    dist = [0.1] * 10
    psi = calculate_psi(dist, dist)
    assert psi == 0.0


def test_psi_slight_shift_is_below_threshold() -> None:
    """A small perturbation should yield a low PSI (< 0.10)."""
    expected = [0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.02]
    # Small random-like shift
    actual = [0.06, 0.07, 0.13, 0.14, 0.17, 0.16, 0.11, 0.09, 0.05, 0.02]

    psi = calculate_psi(expected, actual)
    assert 0.0 <= psi < 0.10, f"Expected minor PSI, got {psi}"


def test_psi_severe_shift_exceeds_warning_threshold() -> None:
    """A major distribution shift (e.g., all low confidences) must trigger high PSI (> 0.20)."""
    # Validation baseline: high confidence skew
    expected = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.10]
    # Severe shift: 90% in lowest bin [0.0, 0.1]
    actual = [0.90, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    psi = calculate_psi(expected, actual)
    assert psi > DEFAULT_PSI_WARNING_THRESHOLD, (
        f"Expected PSI > {DEFAULT_PSI_WARNING_THRESHOLD}, got {psi}"
    )


def test_psi_normalizes_raw_counts() -> None:
    """Raw frequency counts are properly normalized to probabilities."""
    expected_counts = [100, 200, 300, 400]
    actual_counts = [10, 20, 30, 40]  # Exact same proportions (10x smaller)

    psi = calculate_psi(expected_counts, actual_counts)
    assert psi == 0.0


def test_psi_mismatched_bins_raises_value_error() -> None:
    """Mismatched number of bins raises ValueError."""
    with pytest.raises(ValueError, match="Bins count mismatch"):
        calculate_psi([0.5, 0.5], [0.33, 0.33, 0.34])


# ---------------------------------------------------------------------------
# 2. RollingConfidenceBuffer Tests
# ---------------------------------------------------------------------------


def test_buffer_fifo_window_eviction() -> None:
    """Buffer must retain only the most recent window_size items."""
    buf = RollingConfidenceBuffer(window_size=5)
    for i in range(10):
        buf.add(float(i) / 10.0)

    assert len(buf) == 5
    # Should retain 0.5, 0.6, 0.7, 0.8, 0.9
    assert buf.get_scores() == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_buffer_clamping() -> None:
    """Out-of-range scores must be clamped to [0.0, 1.0]."""
    buf = RollingConfidenceBuffer(window_size=10)
    buf.add(-0.5)
    buf.add(1.5)

    scores = buf.get_scores()
    assert scores[0] == 0.0
    assert scores[1] == 1.0


def test_buffer_compute_binned_distribution() -> None:
    """Binned distribution should sum to 1.0 and allocate across correct intervals."""
    buf = RollingConfidenceBuffer(window_size=100)
    # 5 scores in [0.1, 0.2), 5 scores in [0.8, 0.9)
    buf.extend([0.15] * 5 + [0.85] * 5)

    edges, props, counts = buf.compute_binned_distribution(num_bins=10)
    assert len(edges) == 11
    assert len(props) == 10
    assert counts[1] == 5  # bin 1: [0.1, 0.2)
    assert counts[8] == 5  # bin 8: [0.8, 0.9)
    assert sum(props) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. DriftDetector Tests with Fixtures
# ---------------------------------------------------------------------------


def test_drift_status_insufficient_samples() -> None:
    """When sample count < min_samples, status remains 'ok' with nominal flag."""
    detector = DriftDetector(min_samples=20)
    detector.record_confidences([0.05] * 10)  # skewed but only 10 samples

    report = detector.check_drift()
    assert report.status == "ok"
    assert report.min_samples_reached is False
    assert report.sample_count == 10


def test_drift_nominal_status_on_concordant_live_data() -> None:
    """Live scores drawn from validation-like distribution yield status='ok'."""
    detector = DriftDetector(min_samples=50, psi_threshold=0.20)

    # Ingest samples matching the baseline proportions
    samples: list[float] = []
    base_counts = VALIDATION_BASELINE_COUNTS
    total_base = sum(base_counts)
    for idx, count in enumerate(base_counts):
        bin_center = (idx + 0.5) / 10.0
        n_samples = max(2, int((count / total_base) * 200))
        samples.extend([bin_center] * n_samples)

    detector.record_confidences(samples)
    report = detector.check_drift()

    assert report.min_samples_reached is True
    assert report.status == "ok", (
        f"Expected 'ok', got '{report.status}' with PSI={report.psi}"
    )
    assert report.psi < 0.20


def test_drift_warning_status_on_shifted_live_data() -> None:
    """Live scores with heavy distribution shift yield status='warning'."""
    detector = DriftDetector(min_samples=40, psi_threshold=0.20)

    # Ingest 50 samples concentrated heavily in lowest bin
    detector.record_confidences([0.02] * 40 + [0.08] * 10)
    report = detector.check_drift()

    assert report.min_samples_reached is True
    assert report.status == "warning", (
        f"Expected 'warning', got '{report.status}' with PSI={report.psi}"
    )
    assert report.psi >= 0.20
    assert "Confidence drift detected" in report.message


def test_drift_detector_from_calibration_fixture() -> None:
    """Verify loading baseline directly from models/fine-tuned/calibration.json."""
    calib_file = Path("models/fine-tuned/calibration.json")
    if not calib_file.exists():
        pytest.skip("calibration.json fixture not found")

    detector = DriftDetector.from_calibration_file(
        calibration_path=calib_file, min_samples=30
    )
    assert detector.num_bins == 10
    assert sum(detector.baseline_proportions) == pytest.approx(1.0, abs=0.01)

    # Test report serialization
    report = detector.check_drift()
    report_dict = report.to_dict()
    assert "status" in report_dict
    assert "psi" in report_dict
    assert "baseline_distribution" in report_dict
    assert "live_distribution" in report_dict
