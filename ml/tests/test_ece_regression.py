"""
Regression test: calibrated ECE on the fixture must not exceed a stored limit.

This test fails if any code change causes the calibrated ECE to drift above the
limit recorded during Day 3 evaluation (ece_limit = 0.10, i.e. 10 pp). The
fixture is the same 10-sample miscalibrated fixture used in test_calibration.py
and test_ece.py so it is fast and reproducible with no external data.
"""

import numpy as np
import pytest

from ml.eval.calibration import apply_calibration, fit_temperature
from ml.eval.ece import compute_ece

# ---------------------------------------------------------------------------
# Stored limit — Day 3 evaluation
# ECE after temperature scaling on the validation fixture must stay below this.
# Update this constant intentionally when the calibration method changes.
# ---------------------------------------------------------------------------
ECE_LIMIT = 0.10  # 10 pp

# ---------------------------------------------------------------------------
# Shared fixture: 50 "overconfident" predictions with known miscalibration
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
N = 50
# Scores biased toward 0.85–0.99 (overconfident)
SCORES_RAW = RNG.beta(8, 1, size=N).astype(np.float32)
# Only half are true positives — raw ECE is large
IS_TP = (RNG.uniform(size=N) < 0.50).astype(np.float32)


def test_ece_before_calibration_exceeds_limit():
    """Sanity check: raw (pre-calibration) ECE on fixture is above the limit."""
    ece_before = compute_ece(SCORES_RAW, IS_TP, n_bins=10)
    assert ece_before > ECE_LIMIT, (
        f"Expected raw ECE > {ECE_LIMIT:.3f} but got {ece_before:.3f}. "
        "If the fixture changed, verify the test still exercises miscalibration."
    )


def test_calibrated_ece_below_stored_limit():
    """
    Regression guard: calibrated ECE must stay at or below ECE_LIMIT.

    If this test fails, one of the following changed:
      - The temperature scaling fit algorithm (ml/eval/calibration.py)
      - The ECE computation (ml/eval/ece.py)
      - The fixture itself (RNG seed or parameters above)

    Update ECE_LIMIT only if the change is intentional and the new value is
    strictly better (lower).
    """
    temperature = fit_temperature(SCORES_RAW, IS_TP)
    scores_cal = apply_calibration(SCORES_RAW, temperature)
    ece_after = compute_ece(scores_cal, IS_TP, n_bins=10)

    assert ece_after <= ECE_LIMIT, (
        f"Calibrated ECE {ece_after:.4f} exceeds stored limit {ECE_LIMIT:.4f}. "
        "Temperature scaling may have regressed. "
        f"Temperature used: {temperature:.4f}."
    )


def test_calibration_lowers_ece():
    """Temperature scaling must reduce ECE compared to raw scores on the fixture."""
    ece_before = compute_ece(SCORES_RAW, IS_TP, n_bins=10)
    temperature = fit_temperature(SCORES_RAW, IS_TP)
    scores_cal = apply_calibration(SCORES_RAW, temperature)
    ece_after = compute_ece(scores_cal, IS_TP, n_bins=10)

    assert ece_after < ece_before, (
        f"Calibration did not lower ECE: before={ece_before:.4f}, after={ece_after:.4f}."
    )


def test_ece_limit_constant_is_documented():
    """Confirm the ECE_LIMIT module constant has the expected value from Day 3 records."""
    assert ECE_LIMIT == pytest.approx(0.10, abs=1e-9), (
        "ECE_LIMIT changed without a deliberate update. "
        "Change this test only alongside a code improvement that lowered ECE."
    )
