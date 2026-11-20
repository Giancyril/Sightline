"""
ml/eval/calibration.py
-----------------------
Temperature scaling for detector confidence scores (Commit 4).

Temperature scaling is a single-parameter post-hoc calibration method:

    calibrated_score = sigmoid(logit(score) / T)

where T > 0 is a temperature learned by minimising binary cross-entropy
(negative log-likelihood) on the validation set.

  T > 1  → softens over-confident scores  (lower calibrated values)
  T < 1  → sharpens under-confident scores (higher calibrated values)
  T = 1  → no change

The logit/sigmoid pair preserves rank order (monotonic), so AP and
threshold-relative metrics are unaffected.

Public API
----------
fit_temperature(scores, tp_flags, *, bounds, tol, max_iter) -> float
apply_calibration(scores, temperature) -> np.ndarray
save_calibration(path, temperature, meta) -> None
load_calibration(path) -> dict
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

# Clip scores away from 0 and 1 to keep logits finite.
_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise logit."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary cross-entropy loss for a given temperature."""
    scaled = logits / max(temperature, _EPS)
    probs = _sigmoid(scaled)
    probs = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(
        -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    )


def fit_temperature(
    scores: np.ndarray,
    tp_flags: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
    tol: float = 1e-6,
    max_iter: int = 500,
) -> float:
    """
    Fit a temperature scalar T that minimises NLL on (scores, tp_flags).

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Raw detector confidence scores in (0, 1).
    tp_flags : np.ndarray, shape (N,)
        1.0 for true positive, 0.0 for false positive.
    bounds : (float, float)
        Search interval for T.
    tol : float
        Solver tolerance.
    max_iter : int
        Maximum solver iterations.

    Returns
    -------
    float
        Optimal temperature T > 0.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(tp_flags, dtype=np.float64)

    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and tp_flags must have the same shape; got {scores.shape} vs {labels.shape}"
        )
    if len(scores) == 0:
        raise ValueError("scores array must be non-empty")

    logits = _logit(scores)

    result = minimize_scalar(
        _nll,
        args=(logits, labels),
        bounds=bounds,
        method="bounded",
        options={"xatol": tol, "maxiter": max_iter},
    )

    if not result.success and result.fun is None:
        logger.warning("Temperature optimisation did not converge; using T=1.0")
        return 1.0

    temperature = float(result.x)
    nll_before = _nll(1.0, logits, labels)
    nll_after = float(result.fun)
    logger.info(
        "Temperature fit: T=%.4f | NLL %.4f -> %.4f (delta=%.4f)",
        temperature,
        nll_before,
        nll_after,
        nll_before - nll_after,
    )
    return temperature


def apply_calibration(
    scores: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling to raw detector scores.

    Parameters
    ----------
    scores : np.ndarray
        Raw scores in (0, 1).  Any shape is accepted.
    temperature : float
        Temperature T > 0 from :func:`fit_temperature`.

    Returns
    -------
    np.ndarray
        Calibrated scores, same shape as *scores*, values in (0, 1).
        Rank order (monotonicity) is guaranteed because sigmoid and
        division by T > 0 are both monotone functions.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    arr = np.asarray(scores, dtype=np.float64)
    logits = _logit(arr)
    return np.clip(_sigmoid(logits / temperature), _EPS, 1.0 - _EPS).astype(np.float64)


def save_calibration(
    path: str | Path,
    temperature: float,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Persist calibration data as JSON.

    Parameters
    ----------
    path : str or Path
        Output file path.  Parent directories are created automatically.
    temperature : float
        Fitted temperature scalar.
    meta : dict, optional
        Extra metadata to include (e.g. model_id, num_samples, nll_before).
    """
    payload: dict[str, Any] = {"temperature": temperature}
    if meta:
        payload.update(meta)

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved calibration to %s", dest)


def load_calibration(path: str | Path) -> dict[str, Any]:
    """
    Load calibration JSON written by :func:`save_calibration`.

    Returns
    -------
    dict
        At minimum contains ``"temperature"`` key.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If ``"temperature"`` is missing.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Calibration file not found: {src}")

    data: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    if "temperature" not in data:
        raise KeyError(f"'temperature' key missing from {src}")
    return data
