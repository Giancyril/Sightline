"""
ml/eval/ece.py
--------------
Expected Calibration Error (ECE) and reliability diagram (Commit 5).

ECE is defined as the weighted average of |confidence - accuracy| across
equal-width confidence bins:

    ECE = sum_b  (|B_b| / N) * |acc(B_b) - conf(B_b)|

where B_b is the set of predictions in bin b, acc(B_b) is the fraction of
TPs in that bin, and conf(B_b) is the mean confidence in that bin.

A reliability diagram plots acc(B_b) vs conf(B_b).  Perfectly calibrated
predictions lie on the diagonal.

Public API
----------
compute_ece(scores, tp_flags, *, n_bins) -> float
compute_reliability_data(scores, tp_flags, *, n_bins) -> list[BinStat]
save_reliability_json(path, stats_before, stats_after, ece_before, ece_after)
save_reliability_diagram(path, stats_before, stats_after, ece_before, ece_after)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BINS = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BinStat:
    """Statistics for a single confidence bin."""

    bin_lower: float
    bin_upper: float
    mean_confidence: float
    accuracy: float
    count: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_reliability_data(
    scores: np.ndarray,
    tp_flags: np.ndarray,
    *,
    n_bins: int = _DEFAULT_BINS,
) -> list[BinStat]:
    """
    Compute per-bin calibration statistics.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Predicted confidence scores in [0, 1].
    tp_flags : np.ndarray, shape (N,)
        1.0 for true positive, 0.0 for false positive.
    n_bins : int
        Number of equal-width bins covering [0, 1].

    Returns
    -------
    list[BinStat]
        One entry per bin (including empty bins with count=0).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(tp_flags, dtype=np.float64)

    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and tp_flags must have the same shape; got {scores.shape} vs {labels.shape}"
        )

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    stats: list[BinStat] = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Include upper bound in last bin to catch score == 1.0
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)

        cnt = int(np.sum(mask))
        if cnt == 0:
            stats.append(
                BinStat(
                    bin_lower=round(float(lo), 4),
                    bin_upper=round(float(hi), 4),
                    mean_confidence=round(float((lo + hi) / 2), 4),
                    accuracy=0.0,
                    count=0,
                )
            )
        else:
            mean_conf = float(np.mean(scores[mask]))
            acc = float(np.mean(labels[mask]))
            stats.append(
                BinStat(
                    bin_lower=round(float(lo), 4),
                    bin_upper=round(float(hi), 4),
                    mean_confidence=round(mean_conf, 6),
                    accuracy=round(acc, 6),
                    count=cnt,
                )
            )

    return stats


def compute_ece(
    scores: np.ndarray,
    tp_flags: np.ndarray,
    *,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """
    Compute Expected Calibration Error.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Predicted confidence scores in [0, 1].
    tp_flags : np.ndarray, shape (N,)
        1.0 for true positive, 0.0 for false positive.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        ECE in [0, 1].  Lower is better (0 = perfectly calibrated).
    """
    n = len(scores)
    if n == 0:
        return 0.0

    stats = compute_reliability_data(scores, tp_flags, n_bins=n_bins)
    ece = sum(
        (s.count / n) * abs(s.accuracy - s.mean_confidence)
        for s in stats
        if s.count > 0
    )
    return float(ece)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_reliability_json(
    path: str | Path,
    stats_before: list[BinStat],
    stats_after: list[BinStat],
    ece_before: float,
    ece_after: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save reliability diagram data as JSON.

    Parameters
    ----------
    path : str or Path
        Output file path; parent directories are created automatically.
    stats_before : list[BinStat]
        Reliability stats for raw (uncalibrated) scores.
    stats_after : list[BinStat]
        Reliability stats for calibrated scores.
    ece_before / ece_after : float
        ECE values to embed.
    extra : dict, optional
        Any additional fields to include at top level.
    """
    payload: dict[str, Any] = {
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "bins_before": [asdict(s) for s in stats_before],
        "bins_after": [asdict(s) for s in stats_after],
    }
    if extra:
        payload.update(extra)

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved reliability JSON to %s", dest)


def save_reliability_diagram(
    path: str | Path,
    stats_before: list[BinStat],
    stats_after: list[BinStat],
    ece_before: float,
    ece_after: float,
) -> None:
    """
    Render and save a reliability diagram PNG comparing raw vs calibrated scores.

    Parameters
    ----------
    path : str or Path
        Output PNG file path; parent directories are created automatically.
    stats_before / stats_after : list[BinStat]
        Reliability data for the two series.
    ece_before / ece_after : float
        ECE values shown in the legend.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    def _plot_series(
        stats: list[BinStat],
        label: str,
        color: str,
        marker: str,
    ) -> None:
        xs = [s.mean_confidence for s in stats if s.count > 0]
        ys = [s.accuracy for s in stats if s.count > 0]
        ax.plot(
            xs, ys, marker=marker, color=color, label=label, linewidth=1.5, markersize=5
        )

    _plot_series(
        stats_before,
        f"Before calibration (ECE={ece_before:.4f})",
        "#e74c3c",
        "o",
    )
    _plot_series(
        stats_after,
        f"After calibration  (ECE={ece_after:.4f})",
        "#2ecc71",
        "s",
    )

    ax.set_xlabel("Mean confidence", fontsize=11)
    ax.set_ylabel("Accuracy (fraction TP)", fontsize=11)
    ax.set_title("Reliability diagram", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    logger.info("Saved reliability diagram to %s", dest)


def main() -> None:
    import argparse

    from ml.eval.calibration import apply_calibration, fit_temperature
    from ml.eval.dump_predictions import load_predictions_dump
    from ml.eval.precision_recall import compute_per_class_pr

    parser = argparse.ArgumentParser(
        description="Compute ECE and generate reliability diagram."
    )
    parser.add_argument(
        "--dump",
        required=True,
        help="Path to predictions dump .npz.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/calibration",
        help="Directory to save reliability diagram PNG and JSON.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for TP matching.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dump_data = load_predictions_dump(args.dump)
    class_results = compute_per_class_pr(dump_data, iou_threshold=args.iou_thresh)

    all_scores: list[float] = []
    all_tp: list[float] = []
    for cr in class_results.values():
        all_scores.extend(cr.scores.tolist())
        all_tp.extend(cr.tp.tolist())

    scores_arr = np.array(all_scores, dtype=np.float64)
    tp_arr = np.array(all_tp, dtype=np.float64)

    T = fit_temperature(scores_arr, tp_arr)
    cal_scores = apply_calibration(scores_arr, T)

    stats_before = compute_reliability_data(scores_arr, tp_arr)
    stats_after = compute_reliability_data(cal_scores, tp_arr)
    ece_before = compute_ece(scores_arr, tp_arr)
    ece_after = compute_ece(cal_scores, tp_arr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "reliability.json"
    png_path = out_dir / "reliability_diagram.png"

    save_reliability_json(
        json_path,
        stats_before,
        stats_after,
        ece_before,
        ece_after,
        extra={"temperature": round(T, 4)},
    )
    save_reliability_diagram(png_path, stats_before, stats_after, ece_before, ece_after)

    print(f"Reliability JSON: {json_path}")
    print(f"Reliability Diagram: {png_path}")
    print(f"ECE before: {ece_before:.4f} | ECE after: {ece_after:.4f} | T: {T:.4f}")


if __name__ == "__main__":
    main()
