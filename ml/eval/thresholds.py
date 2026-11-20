"""
ml/eval/thresholds.py
---------------------
Per-class confidence threshold selection rules (Commit 3).
Supports Best F1 and Target Precision rules with configurable minimum sample count
and fallback to global threshold.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.eval.dump_predictions import load_predictions_dump
from ml.eval.precision_recall import ClassPRResult, compute_per_class_pr

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration options for per-class threshold selection."""

    rule: str = "best_f1"  # "best_f1" or "target_precision"
    target_precision: float = 0.80
    min_samples: int = 30
    global_threshold: float = 0.25
    iou_threshold: float = 0.50

    @classmethod
    def from_yaml(cls, path_or_str: str | Path) -> ThresholdConfig:
        """Parse configuration from YAML file or string."""
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(path_or_str)
        data = yaml.safe_load(content) or {}
        return cls(
            rule=str(data.get("rule", "best_f1")),
            target_precision=float(data.get("target_precision", 0.80)),
            min_samples=int(data.get("min_samples", 30)),
            global_threshold=float(data.get("global_threshold", 0.25)),
            iou_threshold=float(data.get("iou_threshold", 0.50)),
        )


@dataclass
class ClassThresholdEntry:
    """Selected threshold and resulting validation metrics for a single class."""

    class_id: int
    class_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    num_detections: int
    num_ground_truth: int
    fell_back: bool
    fallback_reason: str | None = None


def _compute_metrics_at_threshold(
    scores: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    num_gt: int,
    thresh: float,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for a given score threshold cutoff."""
    mask = scores >= thresh
    tp_cnt = float(np.sum(tp[mask]))
    fp_cnt = float(np.sum(fp[mask]))
    det_cnt = tp_cnt + fp_cnt

    prec = tp_cnt / det_cnt if det_cnt > 0 else 0.0
    rec = tp_cnt / float(num_gt) if num_gt > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def select_threshold_for_class(
    pr_result: ClassPRResult,
    config: ThresholdConfig,
) -> ClassThresholdEntry:
    """
    Select optimal confidence threshold for a class based on configured rule.
    Falls back to global threshold if validation samples < min_samples or rule fails.
    """
    cname = pr_result.class_name
    cid = pr_result.class_id
    num_det = pr_result.num_det
    num_gt = pr_result.num_gt

    # 1. Fallback: insufficient sample count
    if num_det < config.min_samples:
        reason = (
            f"Insufficient validation detections ({num_det} < {config.min_samples})"
        )
        logger.warning(
            "Class '%s' (id=%d) fell back to global threshold %.2f: %s",
            cname,
            cid,
            config.global_threshold,
            reason,
        )
        p, r, f1 = _compute_metrics_at_threshold(
            pr_result.scores,
            pr_result.tp,
            pr_result.fp,
            num_gt,
            config.global_threshold,
        )
        return ClassThresholdEntry(
            class_id=cid,
            class_name=cname,
            threshold=config.global_threshold,
            precision=p,
            recall=r,
            f1=f1,
            num_detections=num_det,
            num_ground_truth=num_gt,
            fell_back=True,
            fallback_reason=reason,
        )

    scores = pr_result.scores
    precisions = pr_result.precisions
    recalls = pr_result.recalls

    # 2. Rule: best_f1
    if config.rule == "best_f1":
        denom = precisions + recalls
        f1_scores = np.where(denom > 0, (2.0 * precisions * recalls) / denom, 0.0)
        best_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[best_idx])

        if best_f1 <= 0.0:
            reason = "No detections yielded non-zero F1 score"
            logger.warning(
                "Class '%s' fell back to global threshold %.2f: %s",
                cname,
                config.global_threshold,
                reason,
            )
            p, r, f1 = _compute_metrics_at_threshold(
                scores, pr_result.tp, pr_result.fp, num_gt, config.global_threshold
            )
            return ClassThresholdEntry(
                class_id=cid,
                class_name=cname,
                threshold=config.global_threshold,
                precision=p,
                recall=r,
                f1=f1,
                num_detections=num_det,
                num_ground_truth=num_gt,
                fell_back=True,
                fallback_reason=reason,
            )

        chosen_thresh = round(float(scores[best_idx]), 4)
        chosen_prec = float(precisions[best_idx])
        chosen_rec = float(recalls[best_idx])

        return ClassThresholdEntry(
            class_id=cid,
            class_name=cname,
            threshold=chosen_thresh,
            precision=chosen_prec,
            recall=chosen_rec,
            f1=best_f1,
            num_detections=num_det,
            num_ground_truth=num_gt,
            fell_back=False,
            fallback_reason=None,
        )

    # 3. Rule: target_precision (lowest threshold reaching >= target_precision)
    elif config.rule == "target_precision":
        valid_indices = np.where(precisions >= config.target_precision)[0]
        if len(valid_indices) == 0:
            reason = f"Target precision {config.target_precision:.2f} could not be reached (max precision: {float(np.max(precisions)):.2f})"
            logger.warning(
                "Class '%s' fell back to global threshold %.2f: %s",
                cname,
                config.global_threshold,
                reason,
            )
            p, r, f1 = _compute_metrics_at_threshold(
                scores, pr_result.tp, pr_result.fp, num_gt, config.global_threshold
            )
            return ClassThresholdEntry(
                class_id=cid,
                class_name=cname,
                threshold=config.global_threshold,
                precision=p,
                recall=r,
                f1=f1,
                num_detections=num_det,
                num_ground_truth=num_gt,
                fell_back=True,
                fallback_reason=reason,
            )

        # Lowest threshold that satisfies precision target = highest index in sorted-descending scores
        chosen_idx = int(valid_indices[-1])
        chosen_thresh = round(float(scores[chosen_idx]), 4)
        chosen_prec = float(precisions[chosen_idx])
        chosen_rec = float(recalls[chosen_idx])
        chosen_f1 = (
            (2.0 * chosen_prec * chosen_rec) / (chosen_prec + chosen_rec)
            if (chosen_prec + chosen_rec) > 0
            else 0.0
        )

        return ClassThresholdEntry(
            class_id=cid,
            class_name=cname,
            threshold=chosen_thresh,
            precision=chosen_prec,
            recall=chosen_rec,
            f1=chosen_f1,
            num_detections=num_det,
            num_ground_truth=num_gt,
            fell_back=False,
            fallback_reason=None,
        )

    else:
        raise ValueError(f"Unknown threshold selection rule: '{config.rule}'")


def select_all_thresholds(
    pr_results: dict[str, ClassPRResult],
    config: ThresholdConfig | dict[str, Any] | None = None,
) -> list[ClassThresholdEntry]:
    """Select thresholds for all classes in PR results."""
    if config is None:
        cfg = ThresholdConfig()
    elif isinstance(config, dict):
        cfg = ThresholdConfig(**config)
    else:
        cfg = config

    entries = [
        select_threshold_for_class(res, cfg)
        for _, res in sorted(pr_results.items(), key=lambda item: item[1].class_id)
    ]
    return entries


def format_thresholds_table(entries: list[ClassThresholdEntry]) -> str:
    """Format per-class threshold entries into an aligned ASCII table."""
    lines = [
        "=" * 88,
        f"{'ID':<4} {'Class Name':<18} {'Threshold':<11} {'Precision':<11} {'Recall':<10} {'F1':<8} {'Detections':<12} {'Status'}",
        "=" * 88,
    ]
    for e in entries:
        status = "Fallback" if e.fell_back else "Optimal"
        lines.append(
            f"{e.class_id:<4} {e.class_name:<18} {e.threshold:<11.4f} {e.precision:<11.4f} {e.recall:<10.4f} {e.f1:<8.4f} {e.num_detections:<12} {status}"
        )
    lines.append("=" * 88)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select per-class confidence thresholds."
    )
    parser.add_argument("--dump", required=True, help="Path to predictions dump .npz.")
    parser.add_argument(
        "--rule", default="best_f1", choices=["best_f1", "target_precision"]
    )
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--global-threshold", type=float, default=0.25)
    parser.add_argument(
        "--config-yaml", default=None, help="Optional YAML config file."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.config_yaml:
        cfg = ThresholdConfig.from_yaml(args.config_yaml)
    else:
        cfg = ThresholdConfig(
            rule=args.rule,
            target_precision=args.target_precision,
            min_samples=args.min_samples,
            global_threshold=args.global_threshold,
        )

    dump_data = load_predictions_dump(args.dump)
    pr_results = compute_per_class_pr(dump_data, iou_threshold=cfg.iou_threshold)
    entries = select_all_thresholds(pr_results, cfg)

    print("\n" + format_thresholds_table(entries) + "\n")


if __name__ == "__main__":
    main()
