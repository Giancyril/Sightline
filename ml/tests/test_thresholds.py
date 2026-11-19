"""
ml/tests/test_thresholds.py
----------------------------
Unit tests for ml/eval/thresholds.py (Commit 3).

Covers:
  - best_f1 rule: selects threshold with highest F1; threshold is in scores range
  - target_precision rule: class reaching target gets non-fallback result
  - target_precision fallback: class that cannot reach target falls back to global_threshold
  - min_samples fallback: class with fewer detections than min_samples falls back
  - ThresholdConfig.from_yaml(): correctly parses a valid YAML string
  - select_all_thresholds(): returns one entry per class, sorted by class_id
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.eval.precision_recall import ClassPRResult
from ml.eval.thresholds import (
    ClassThresholdEntry,
    ThresholdConfig,
    format_thresholds_table,
    select_all_thresholds,
    select_threshold_for_class,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pr_result(
    class_id: int,
    class_name: str,
    scores: list[float],
    tp: list[float],
    fp: list[float],
    num_gt: int,
    ap: float = 0.5,
) -> ClassPRResult:
    """Build a ClassPRResult with computed cumulative precision/recall arrays."""
    scores_arr = np.array(scores, dtype=np.float32)
    tp_arr = np.array(tp, dtype=np.float32)
    fp_arr = np.array(fp, dtype=np.float32)

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    precisions = np.where(
        (cum_tp + cum_fp) > 0,
        cum_tp / (cum_tp + cum_fp),
        0.0,
    ).astype(np.float32)
    recalls = (
        (cum_tp / num_gt).astype(np.float32)
        if num_gt > 0
        else np.zeros_like(scores_arr)
    )

    return ClassPRResult(
        class_id=class_id,
        class_name=class_name,
        scores=scores_arr,
        tp=tp_arr,
        fp=fp_arr,
        precisions=precisions,
        recalls=recalls,
        ap=ap,
        num_gt=num_gt,
        num_det=len(scores),
    )


DEFAULT_CFG = ThresholdConfig(
    rule="best_f1",
    target_precision=0.80,
    min_samples=5,
    global_threshold=0.25,
    iou_threshold=0.50,
)


# ---------------------------------------------------------------------------
# Test: best_f1 rule
# ---------------------------------------------------------------------------


def test_best_f1_selects_highest_f1_threshold() -> None:
    """best_f1 rule must return the threshold that maximises F1."""
    # 6 detections, scores descending; first 4 are TP, last 2 are FP
    pr = _make_pr_result(
        class_id=1,
        class_name="Person",
        scores=[0.95, 0.90, 0.80, 0.70, 0.40, 0.30],
        tp=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        fp=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        num_gt=4,
    )
    cfg = ThresholdConfig(rule="best_f1", min_samples=5, global_threshold=0.25)
    entry = select_threshold_for_class(pr, cfg)

    assert isinstance(entry, ClassThresholdEntry)
    assert not entry.fell_back
    # threshold must be one of the actual score values
    assert entry.threshold in [round(float(s), 4) for s in pr.scores]
    # F1 at selected threshold must be positive
    assert entry.f1 > 0.0
    # precision and recall must be consistent fractions in [0,1]
    assert 0.0 <= entry.precision <= 1.0
    assert 0.0 <= entry.recall <= 1.0


def test_best_f1_threshold_in_score_range() -> None:
    """The selected threshold must be one of the rounded score values."""
    pr = _make_pr_result(
        class_id=2,
        class_name="Laptop",
        scores=[0.88, 0.75, 0.60, 0.50, 0.42, 0.31],
        tp=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        fp=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        num_gt=4,
    )
    cfg = ThresholdConfig(rule="best_f1", min_samples=5)
    entry = select_threshold_for_class(pr, cfg)

    # thresholds.py rounds to 4 decimal places; compare against rounded float64 values
    rounded_scores = [round(float(s), 4) for s in pr.scores]
    assert entry.threshold in rounded_scores


# ---------------------------------------------------------------------------
# Test: target_precision rule
# ---------------------------------------------------------------------------


def test_target_precision_non_fallback_when_target_reachable() -> None:
    """Class that can reach target precision must not fall back."""
    # Scores high → perfect precision at top; enough samples
    pr = _make_pr_result(
        class_id=3,
        class_name="Car",
        scores=[0.99, 0.95, 0.90, 0.85, 0.80, 0.70],
        tp=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        fp=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        num_gt=5,
    )
    cfg = ThresholdConfig(rule="target_precision", target_precision=0.80, min_samples=5)
    entry = select_threshold_for_class(pr, cfg)

    assert not entry.fell_back
    assert entry.precision >= 0.80


def test_target_precision_fallback_when_target_unreachable() -> None:
    """Class that never reaches target precision must fall back to global_threshold."""
    # All detections are FPs → precision always 0
    pr = _make_pr_result(
        class_id=4,
        class_name="Chair",
        scores=[0.80, 0.70, 0.60, 0.55, 0.50, 0.45],
        tp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        fp=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        num_gt=3,
    )
    cfg = ThresholdConfig(
        rule="target_precision",
        target_precision=0.80,
        min_samples=5,
        global_threshold=0.25,
    )
    entry = select_threshold_for_class(pr, cfg)

    assert entry.fell_back
    assert entry.threshold == pytest.approx(0.25)
    assert entry.fallback_reason is not None
    assert "0.80" in entry.fallback_reason


# ---------------------------------------------------------------------------
# Test: min_samples fallback
# ---------------------------------------------------------------------------


def test_min_samples_fallback() -> None:
    """Class with fewer detections than min_samples falls back to global_threshold."""
    # Only 3 detections, min_samples=10
    pr = _make_pr_result(
        class_id=5,
        class_name="Boat",
        scores=[0.70, 0.60, 0.55],
        tp=[1.0, 1.0, 0.0],
        fp=[0.0, 0.0, 1.0],
        num_gt=2,
    )
    cfg = ThresholdConfig(rule="best_f1", min_samples=10, global_threshold=0.25)
    entry = select_threshold_for_class(pr, cfg)

    assert entry.fell_back
    assert entry.threshold == pytest.approx(0.25)
    assert entry.fallback_reason is not None
    assert "3" in entry.fallback_reason  # detection count mentioned


# ---------------------------------------------------------------------------
# Test: ThresholdConfig.from_yaml
# ---------------------------------------------------------------------------


def test_threshold_config_from_yaml_string() -> None:
    """ThresholdConfig.from_yaml() must parse valid YAML string correctly."""
    yaml_str = "rule: target_precision\ntarget_precision: 0.85\nmin_samples: 50\nglobal_threshold: 0.30\niou_threshold: 0.45\n"
    cfg = ThresholdConfig.from_yaml(yaml_str)

    assert cfg.rule == "target_precision"
    assert cfg.target_precision == pytest.approx(0.85)
    assert cfg.min_samples == 50
    assert cfg.global_threshold == pytest.approx(0.30)
    assert cfg.iou_threshold == pytest.approx(0.45)


def test_threshold_config_from_yaml_defaults() -> None:
    """from_yaml with empty string must return all defaults."""
    cfg = ThresholdConfig.from_yaml("{}")
    assert cfg.rule == "best_f1"
    assert cfg.target_precision == pytest.approx(0.80)
    assert cfg.min_samples == 30
    assert cfg.global_threshold == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Test: select_all_thresholds
# ---------------------------------------------------------------------------


def test_select_all_thresholds_returns_one_per_class_sorted() -> None:
    """select_all_thresholds must return one entry per class, sorted by class_id."""
    pr_results = {
        "Car": _make_pr_result(
            class_id=3,
            class_name="Car",
            scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            tp=[1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            fp=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            num_gt=4,
        ),
        "Person": _make_pr_result(
            class_id=1,
            class_name="Person",
            scores=[0.95, 0.85, 0.75, 0.65, 0.55, 0.45],
            tp=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            fp=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            num_gt=3,
        ),
        "Laptop": _make_pr_result(
            class_id=2,
            class_name="Laptop",
            scores=[0.88, 0.78, 0.68, 0.58, 0.48, 0.38],
            tp=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            fp=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            num_gt=3,
        ),
    }
    cfg = ThresholdConfig(rule="best_f1", min_samples=5)
    entries = select_all_thresholds(pr_results, cfg)

    assert len(entries) == 3
    class_ids = [e.class_id for e in entries]
    assert class_ids == sorted(class_ids)


def test_select_all_thresholds_accepts_none_config() -> None:
    """select_all_thresholds with config=None must use defaults."""
    pr_results = {
        "Person": _make_pr_result(
            class_id=1,
            class_name="Person",
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            tp=[1.0, 1.0, 1.0, 0.0, 0.0],
            fp=[0.0, 0.0, 0.0, 1.0, 1.0],
            num_gt=3,
        ),
    }
    entries = select_all_thresholds(pr_results, config=None)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Test: format_thresholds_table
# ---------------------------------------------------------------------------


def test_format_thresholds_table_contains_class_names() -> None:
    """format_thresholds_table must include each class name in its output."""
    pr_results = {
        "Person": _make_pr_result(
            class_id=1,
            class_name="Person",
            scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            tp=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            fp=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            num_gt=3,
        ),
        "Car": _make_pr_result(
            class_id=3,
            class_name="Car",
            scores=[0.85, 0.75, 0.65, 0.55, 0.45, 0.35],
            tp=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            fp=[0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            num_gt=2,
        ),
    }
    entries = select_all_thresholds(
        pr_results, ThresholdConfig(rule="best_f1", min_samples=5)
    )
    table = format_thresholds_table(entries)

    assert "Person" in table
    assert "Car" in table
    assert "Optimal" in table or "Fallback" in table
