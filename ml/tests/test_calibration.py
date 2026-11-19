"""
ml/tests/test_calibration.py
-----------------------------
Unit tests for ml/eval/calibration.py (Commit 4).

Covers:
  - fit_temperature reduces NLL vs T=1 on a known miscalibrated fixture
  - apply_calibration preserves rank order (monotonicity)
  - apply_calibration is a no-op when T=1 (up to float tolerance)
  - over-confident scores (high raw, mixed TP/FP) produce T > 1
  - under-confident scores produce T < 1
  - save_calibration / load_calibration round-trip
  - error paths: empty scores, mismatched shapes, T<=0, missing file, missing key
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.eval.calibration import (
    apply_calibration,
    fit_temperature,
    load_calibration,
    save_calibration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# Over-confident fixture: scores are clustered near 1.0 but many are FP.
# A well-calibrated model would assign lower scores to FPs.
_N = 200
_TRUE_PROB = 0.5  # 50 % TP rate


def _make_overconfident_fixture() -> tuple[np.ndarray, np.ndarray]:
    """High raw scores but only ~50 % are TP → model is overconfident."""
    scores = RNG.uniform(0.75, 0.99, _N)
    tp_flags = RNG.binomial(1, _TRUE_PROB, _N).astype(np.float64)
    return scores, tp_flags


def _make_underconfident_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Low raw scores but ~90 % are TP → model is underconfident."""
    scores = RNG.uniform(0.01, 0.30, _N)
    tp_flags = RNG.binomial(1, 0.90, _N).astype(np.float64)
    return scores, tp_flags


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_reduces_nll() -> None:
    """NLL after calibration must be strictly lower than NLL at T=1."""
    scores, tp_flags = _make_overconfident_fixture()
    T = fit_temperature(scores, tp_flags)
    assert T > 0

    from ml.eval.calibration import _logit, _nll  # private helpers

    logits = _logit(np.asarray(scores, dtype=np.float64))
    labels = np.asarray(tp_flags, dtype=np.float64)
    nll_before = _nll(1.0, logits, labels)
    nll_after = _nll(T, logits, labels)
    assert nll_after < nll_before, (
        f"Expected calibrated NLL ({nll_after:.4f}) < uncalibrated ({nll_before:.4f})"
    )


def test_fit_temperature_overconfident_gives_T_gt_1() -> None:
    """Over-confident detector (high scores, many FPs) should yield T > 1."""
    scores, tp_flags = _make_overconfident_fixture()
    T = fit_temperature(scores, tp_flags)
    assert T > 1.0, f"Expected T > 1 for over-confident detector; got T={T:.4f}"


def test_fit_temperature_underconfident_gives_T_lt_1() -> None:
    """
    For a detector whose scores are genuinely well-spread and under-confident,
    temperature scaling should yield T < 1 to amplify the logits.

    We construct a fixture where scores span (0.3, 0.9) and TP rate follows
    the score (high score → more likely TP), so the model is merely
    slightly under-confident — the optimal T should be < 1.
    """
    rng = np.random.default_rng(0)
    n = 300
    scores = rng.uniform(0.30, 0.90, n)
    # TP probability = score + 0.1 (capped at 1) → model under-states confidence
    tp_prob = np.clip(scores + 0.15, 0.0, 1.0)
    tp_flags = rng.binomial(1, tp_prob).astype(np.float64)
    T = fit_temperature(scores, tp_flags)
    assert T < 1.0, f"Expected T < 1 for under-confident detector; got T={T:.4f}"


def test_fit_temperature_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_temperature(np.array([]), np.array([]))


def test_fit_temperature_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        fit_temperature(np.array([0.8, 0.6]), np.array([1.0]))


# ---------------------------------------------------------------------------
# apply_calibration
# ---------------------------------------------------------------------------


def test_apply_calibration_monotone() -> None:
    """Calibration must preserve rank order (monotonicity)."""
    scores = np.linspace(0.01, 0.99, 100)
    for T in (0.5, 1.0, 2.0, 5.0):
        cal = apply_calibration(scores, T)
        diffs = np.diff(cal)
        assert np.all(diffs > 0), f"Monotonicity violated at T={T}"


def test_apply_calibration_identity_at_T1() -> None:
    """At T=1 calibrated scores equal raw scores (up to float tolerance)."""
    scores = np.linspace(0.05, 0.95, 50)
    cal = apply_calibration(scores, 1.0)
    np.testing.assert_allclose(cal, scores, atol=1e-6)


def test_apply_calibration_T_gt_1_lowers_scores() -> None:
    """T > 1 (over-confident) must shift scores toward 0.5 / lower end."""
    scores = np.array([0.9, 0.8, 0.7])
    cal = apply_calibration(scores, 2.0)
    assert np.all(cal < scores), "T>1 should lower high scores"


def test_apply_calibration_T_lt_1_sharpens_scores() -> None:
    """
    T < 1 amplifies logits: scores near 0.5 diverge away from 0.5
    (scores above 0.5 go higher, scores below 0.5 go lower).
    """
    # Above 0.5: T<1 should push scores higher
    high_scores = np.array([0.6, 0.7, 0.8])
    cal_high = apply_calibration(high_scores, 0.5)
    assert np.all(cal_high > high_scores), "T<1 should raise scores above 0.5"

    # Below 0.5: T<1 should push scores lower
    low_scores = np.array([0.2, 0.3, 0.4])
    cal_low = apply_calibration(low_scores, 0.5)
    assert np.all(cal_low < low_scores), "T<1 should lower scores below 0.5"


def test_apply_calibration_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_calibration(np.array([0.5]), 0.0)

    with pytest.raises(ValueError, match="positive"):
        apply_calibration(np.array([0.5]), -1.0)


def test_apply_calibration_output_in_01() -> None:
    """Output must always be strictly in (0, 1) — no exact 0 or 1."""
    scores = np.linspace(0.001, 0.999, 200)
    # Test a range of reasonable temperatures (avoid extreme T like 0.1 that
    # saturates scores to 0/1 by design)
    for T in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
        cal = apply_calibration(scores, T)
        assert np.all(cal > 0) and np.all(cal < 1), f"Out-of-range values at T={T}"


# ---------------------------------------------------------------------------
# save_calibration / load_calibration
# ---------------------------------------------------------------------------


def test_save_load_round_trip() -> None:
    """save_calibration then load_calibration must recover the temperature."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "calibration.json"
        save_calibration(
            path, temperature=1.42, meta={"model_id": "fine-tuned", "n": 500}
        )
        data = load_calibration(path)

    assert data["temperature"] == pytest.approx(1.42)
    assert data["model_id"] == "fine-tuned"
    assert data["n"] == 500


def test_load_calibration_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_calibration("/nonexistent/path/calibration.json")


def test_load_calibration_missing_temperature_key_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.json"
        path.write_text(json.dumps({"model_id": "test"}), encoding="utf-8")
        with pytest.raises(KeyError, match="temperature"):
            load_calibration(path)


def test_save_creates_parent_dirs() -> None:
    """save_calibration must create nested parent directories automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "a" / "b" / "c" / "cal.json"
        save_calibration(path, temperature=0.9)
        assert path.exists()
