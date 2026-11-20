"""
ml/tests/test_ece.py
---------------------
Unit tests for ml/eval/ece.py (Commit 5).

Covers:
  - perfectly calibrated fixture → ECE ≈ 0
  - maximally miscalibrated fixture → ECE close to 1
  - ECE decreases after temperature calibration
  - compute_reliability_data: bin count totals equal N, all bins accounted for
  - save_reliability_json / load round-trip: ece_before/after and bin counts match
  - save_reliability_diagram: file exists and is a valid PNG
  - error paths: mismatched shapes
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.eval.calibration import apply_calibration, fit_temperature
from ml.eval.ece import (
    BinStat,
    compute_ece,
    compute_reliability_data,
    save_reliability_diagram,
    save_reliability_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)


def _perfectly_calibrated(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Scores uniformly in (0,1); TP probability = score.
    This is the definition of perfect calibration: acc(bin) ≈ conf(bin).
    """
    scores = RNG.uniform(0.01, 0.99, n)
    tp_flags = RNG.binomial(1, scores).astype(np.float64)
    return scores, tp_flags


def _maximally_miscalibrated(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    High confidence scores but all FPs → acc=0 in every bin.
    ECE ≈ mean_confidence (close to 1 for high-score fixture).
    """
    scores = RNG.uniform(0.80, 0.99, n)
    tp_flags = np.zeros(n, dtype=np.float64)
    return scores, tp_flags


def _overconfident(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """High scores, only 50 % TP — classic over-confident detector."""
    scores = RNG.uniform(0.75, 0.99, n)
    tp_flags = RNG.binomial(1, 0.50, n).astype(np.float64)
    return scores, tp_flags


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_is_small() -> None:
    """ECE of a perfectly calibrated detector should be close to 0."""
    scores, tp_flags = _perfectly_calibrated(n=2000)
    ece = compute_ece(scores, tp_flags, n_bins=10)
    assert ece < 0.05, f"Expected ECE < 0.05 for perfect calibration; got {ece:.4f}"


def test_ece_maximally_miscalibrated_is_large() -> None:
    """ECE of maximally miscalibrated detector should be close to mean confidence."""
    scores, tp_flags = _maximally_miscalibrated(n=1000)
    ece = compute_ece(scores, tp_flags, n_bins=10)
    expected_min = float(np.mean(scores)) * 0.8  # generous lower bound
    assert ece > expected_min, f"Expected ECE > {expected_min:.4f}; got {ece:.4f}"


def test_ece_in_01_range() -> None:
    """ECE must always be in [0, 1]."""
    for _ in range(5):
        scores = RNG.uniform(0.01, 0.99, 300)
        tp_flags = RNG.binomial(1, 0.5, 300).astype(np.float64)
        ece = compute_ece(scores, tp_flags)
        assert 0.0 <= ece <= 1.0


def test_ece_empty_input_returns_zero() -> None:
    assert compute_ece(np.array([]), np.array([])) == 0.0


def test_ece_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_ece(np.array([0.5, 0.6]), np.array([1.0]))


def test_ece_decreases_after_calibration() -> None:
    """ECE must strictly decrease when temperature calibration is applied."""
    scores, tp_flags = _overconfident(n=600)
    ece_before = compute_ece(scores, tp_flags, n_bins=10)
    T = fit_temperature(scores, tp_flags)
    cal_scores = apply_calibration(scores, T)
    ece_after = compute_ece(cal_scores, tp_flags, n_bins=10)
    assert ece_after < ece_before, (
        f"ECE should decrease after calibration; before={ece_before:.4f}, after={ece_after:.4f}"
    )


# ---------------------------------------------------------------------------
# compute_reliability_data
# ---------------------------------------------------------------------------


def test_reliability_data_bin_counts_sum_to_n() -> None:
    """Total count across all bins must equal len(scores)."""
    scores, tp_flags = _perfectly_calibrated(n=300)
    stats = compute_reliability_data(scores, tp_flags, n_bins=10)
    assert len(stats) == 10
    assert sum(s.count for s in stats) == 300


def test_reliability_data_accuracy_in_01() -> None:
    """accuracy of every non-empty bin must be in [0, 1]."""
    scores, tp_flags = _overconfident(n=400)
    stats = compute_reliability_data(scores, tp_flags, n_bins=10)
    for s in stats:
        if s.count > 0:
            assert 0.0 <= s.accuracy <= 1.0, f"accuracy out of range: {s}"


def test_reliability_data_bin_boundaries() -> None:
    """bin_lower and bin_upper must tile [0, 1] with correct width."""
    stats = compute_reliability_data(np.array([0.5]), np.array([1.0]), n_bins=5)
    assert len(stats) == 5
    assert stats[0].bin_lower == pytest.approx(0.0, abs=1e-4)
    assert stats[-1].bin_upper == pytest.approx(1.0, abs=1e-4)
    for s in stats:
        assert s.bin_upper > s.bin_lower


def test_reliability_data_returns_bin_stat_instances() -> None:
    scores = np.linspace(0.1, 0.9, 50)
    tp_flags = np.ones(50)
    stats = compute_reliability_data(scores, tp_flags, n_bins=10)
    assert all(isinstance(s, BinStat) for s in stats)


# ---------------------------------------------------------------------------
# save_reliability_json
# ---------------------------------------------------------------------------


def test_save_reliability_json_round_trip() -> None:
    """JSON must include ece_before/after and correct bin counts."""
    scores, tp_flags = _overconfident(n=300)
    T = fit_temperature(scores, tp_flags)
    cal = apply_calibration(scores, T)

    stats_before = compute_reliability_data(scores, tp_flags)
    stats_after = compute_reliability_data(cal, tp_flags)
    ece_b = compute_ece(scores, tp_flags)
    ece_a = compute_ece(cal, tp_flags)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "reliability.json"
        save_reliability_json(path, stats_before, stats_after, ece_b, ece_a)
        data = json.loads(path.read_text())

    assert "ece_before" in data
    assert "ece_after" in data
    assert data["ece_before"] == pytest.approx(ece_b, abs=1e-5)
    assert data["ece_after"] == pytest.approx(ece_a, abs=1e-5)
    assert len(data["bins_before"]) == 10
    assert len(data["bins_after"]) == 10
    assert sum(b["count"] for b in data["bins_before"]) == 300


def test_save_reliability_json_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "a" / "b" / "rel.json"
        save_reliability_json(path, [], [], 0.1, 0.05)
        assert path.exists()


# ---------------------------------------------------------------------------
# save_reliability_diagram
# ---------------------------------------------------------------------------


def test_save_reliability_diagram_creates_png() -> None:
    """Diagram file must exist and start with the PNG magic bytes."""
    scores, tp_flags = _overconfident(n=200)
    T = fit_temperature(scores, tp_flags)
    cal = apply_calibration(scores, T)

    stats_before = compute_reliability_data(scores, tp_flags)
    stats_after = compute_reliability_data(cal, tp_flags)
    ece_b = compute_ece(scores, tp_flags)
    ece_a = compute_ece(cal, tp_flags)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "reliability.png"
        save_reliability_diagram(path, stats_before, stats_after, ece_b, ece_a)
        assert path.exists()
        magic = path.read_bytes()[:8]

    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    assert magic == b"\x89PNG\r\n\x1a\n", "File does not start with PNG magic bytes"


def test_save_reliability_diagram_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "dir" / "rel.png"
        save_reliability_diagram(path, [], [], 0.2, 0.1)
        assert path.exists()
