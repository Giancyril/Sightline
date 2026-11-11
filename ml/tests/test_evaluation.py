"""
ml/tests/test_evaluation.py
---------------------------
Stage 5 unit tests: precision-recall calculation, COCO mAP range,
diagnostic error categorization, threshold tuning, success gates verification,
and exported model specification validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.eval.evaluate import run_full_evaluation
from ml.eval.metrics import (
    categorize_detection_errors,
    compute_coco_map_range,
    compute_precision_recall_curve,
    tune_thresholds,
)
from ml.export.export import verify_exported_model

pytestmark = pytest.mark.stage5

ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Precision-Recall & AP Tests
# ---------------------------------------------------------------------------


def test_compute_precision_recall_perfect_predictions():
    """Perfect matching predictions must yield AP = 1.0."""
    gt_boxes = [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]
    pred_boxes = [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]
    pred_scores = [0.99, 0.95]

    recalls, precisions, ap = compute_precision_recall_curve(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
    )

    assert abs(ap - 1.0) < 1e-4, f"Expected AP = 1.0, got {ap}"
    assert len(recalls) == 2
    assert recalls[-1] == 1.0
    assert precisions[-1] == 1.0


def test_compute_precision_recall_no_predictions():
    """Zero predictions must yield AP = 0.0 and recall = 0.0."""
    gt_boxes = [[0.1, 0.1, 0.5, 0.5]]
    recalls, precisions, ap = compute_precision_recall_curve([], [], gt_boxes)

    assert ap == 0.0
    assert recalls[0] == 0.0


def test_compute_coco_map_range():
    """COCO mAP@[0.5:0.95] should be in [0, 1] and <= mAP@0.5."""
    gt_boxes = [[0.1, 0.1, 0.5, 0.5]]
    # Box slightly shifted so IoU degrades at higher thresholds
    pred_boxes = [[0.12, 0.12, 0.52, 0.52]]
    pred_scores = [0.9]

    _, _, ap_50 = compute_precision_recall_curve(pred_boxes, pred_scores, gt_boxes, 0.5)
    coco_map = compute_coco_map_range(pred_boxes, pred_scores, gt_boxes)

    assert 0.0 <= coco_map <= 1.0
    assert coco_map <= ap_50 + 1e-6, "COCO mAP range should be <= AP@0.5"


# ---------------------------------------------------------------------------
# Error Categorization Tests
# ---------------------------------------------------------------------------


def test_categorize_detection_errors():
    """Verify classification of TP, Loc Error, Class Confusion, Bg Confusion, FN."""
    gt_boxes = [
        [0.1, 0.1, 0.4, 0.4],  # Class 1
        [0.5, 0.5, 0.9, 0.9],  # Class 2
    ]
    gt_classes = [1, 2]

    pred_boxes = [
        [0.1, 0.1, 0.4, 0.4],  # TP: class 1, exact match
        [0.15, 0.15, 0.45, 0.45],  # Loc error: class 1, IoU between 0.1 and 0.5
        [0.5, 0.5, 0.9, 0.9],  # Class confusion: matches GT 2 but predicts class 3
        [0.0, 0.8, 0.2, 0.95],  # Background confusion: no overlap with any GT
    ]
    pred_scores = [0.95, 0.85, 0.80, 0.70]
    pred_classes = [1, 1, 3, 1]

    errors = categorize_detection_errors(
        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5
    )

    assert errors["true_positives"] == 1
    assert errors["loc_errors"] >= 1
    assert errors["class_confusion"] >= 1
    assert errors["background_confusion"] >= 1


# ---------------------------------------------------------------------------
# Threshold Tuning Tests
# ---------------------------------------------------------------------------


def test_tune_thresholds():
    """Verify threshold sweep selects optimal F1 balance."""
    candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    precisions = [0.4, 0.6, 0.8, 0.9, 0.95]
    recalls = [0.95, 0.85, 0.75, 0.5, 0.2]

    res = tune_thresholds(candidates, precisions, recalls)

    assert "best_threshold" in res
    assert "best_f1" in res
    assert res["best_f1"] > 0.0
    assert res["best_threshold"] in candidates
    assert len(res["sweep_table"]) == len(candidates)


# ---------------------------------------------------------------------------
# Success Gates Verification
# ---------------------------------------------------------------------------


def test_fine_tuned_success_gates():
    """Verify Stage 5 success criteria from docs/fine_tuned_metrics.json."""
    metrics_path = ROOT / "docs" / "fine_tuned_metrics.json"
    if not metrics_path.exists():
        run_full_evaluation()

    assert metrics_path.exists(), "fine_tuned_metrics.json must be generated"
    data = json.loads(metrics_path.read_text())

    gates = data["success_gates"]
    assert gates["novel_classes_passed"] is True, (
        f"Novel classes target (+10 pts) failed: actual gain = {gates['novel_classes_gain_actual']}"
    )
    assert gates["coco_shared_passed"] is True, (
        f"COCO shared classes retention failed: actual delta = {gates['coco_shared_drop_actual']}"
    )
    assert gates["all_gates_passed"] is True


def test_evaluation_markdown_report_content():
    """Verify docs/evaluation.md contains complete analysis sections."""
    doc_path = ROOT / "docs" / "evaluation.md"
    assert doc_path.exists(), "evaluation.md must exist"

    content = doc_path.read_text()
    assert "Executive Summary & Success Gates" in content
    assert "Per-Class AP Comparison" in content
    assert "Confidence Threshold Tuning" in content
    assert "Diagnostic Error Analysis" in content
    assert "Novel Classes mAP@0.5" in content
    assert "COCO Shared Classes mAP@0.5" in content


# ---------------------------------------------------------------------------
# Model Export & Verification Tests
# ---------------------------------------------------------------------------


def test_exported_model_spec():
    """Verify exported model specification adheres to deployment requirements."""
    spec_path = ROOT / "ml" / "export" / "model_spec.json"
    assert spec_path.exists(), "model_spec.json must exist"

    spec = json.loads(spec_path.read_text())
    assert spec["input_shape"] == [1, 3, 300, 300]
    assert spec["locs_output_shape"] == [1, 2268, 4]
    assert spec["scores_output_shape"] == [1, 2268, 21]
    assert spec["num_classes"] == 21
    assert spec["num_anchors"] == 2268
    assert spec["cpu_latency_ms"] < 300.0, (
        f"Latency {spec['cpu_latency_ms']} ms exceeds 300 ms SLA"
    )
    assert verify_exported_model(spec_path) is True
