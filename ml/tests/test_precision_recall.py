"""
ml/tests/test_precision_recall.py
---------------------------------
Unit tests for per-class precision-recall computation and AP metrics (Commit 2).
Includes toy sets with analytical ground-truth solutions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.eval.dump_predictions import load_predictions_dump
from ml.eval.precision_recall import (
    calculate_ap,
    compute_ap_summary,
    compute_per_class_pr,
    mean_average_precision,
)

FIXTURE_PATH = Path("ml/tests/fixtures/val_dump_fixture.npz")


def test_perfect_detector():
    """
    Toy set 1: Perfect detector.
    2 ground-truth Cars across 2 images.
    2 detections matching each GT with high IoU and confidence.
    Analytical result: Precision = 1.0, Recall = 1.0, AP = 1.0.
    """
    dump_data = {
        "detections": {
            "image_ids": np.array(["img_1", "img_2"]),
            "class_ids": np.array([9, 9], dtype=np.int32),
            "class_names": np.array(["Car", "Car"]),
            "scores": np.array([0.95, 0.88], dtype=np.float32),
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.50, 0.50],
                    [0.20, 0.20, 0.60, 0.60],
                ],
                dtype=np.float32,
            ),
        },
        "ground_truth": {
            "image_ids": np.array(["img_1", "img_2"]),
            "class_ids": np.array([9, 9], dtype=np.int32),
            "class_names": np.array(["Car", "Car"]),
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.50, 0.50],
                    [0.20, 0.20, 0.60, 0.60],
                ],
                dtype=np.float32,
            ),
        },
    }

    res = compute_per_class_pr(dump_data, iou_threshold=0.5)
    assert "Car" in res
    car_res = res["Car"]

    assert car_res.num_gt == 2
    assert car_res.num_det == 2
    assert np.array_equal(car_res.tp, [1.0, 1.0])
    assert np.array_equal(car_res.fp, [0.0, 0.0])
    assert np.allclose(car_res.recalls, [0.5, 1.0])
    assert np.allclose(car_res.precisions, [1.0, 1.0])
    assert pytest.approx(car_res.ap, abs=1e-5) == 1.0


def test_one_missed_object():
    """
    Toy set 2: One missed object.
    2 ground-truth Laptops on 1 image.
    Only 1 detection (matching GT 1). GT 2 is missed.
    Analytical result: Max recall = 0.5, Precision = 1.0, AP = 0.5.
    """
    dump_data = {
        "detections": {
            "image_ids": np.array(["img_1"]),
            "class_ids": np.array([2], dtype=np.int32),
            "class_names": np.array(["Laptop"]),
            "scores": np.array([0.90], dtype=np.float32),
            "boxes": np.array([[0.10, 0.10, 0.40, 0.40]], dtype=np.float32),
        },
        "ground_truth": {
            "image_ids": np.array(["img_1", "img_1"]),
            "class_ids": np.array([2, 2], dtype=np.int32),
            "class_names": np.array(["Laptop", "Laptop"]),
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.60, 0.60, 0.90, 0.90],
                ],
                dtype=np.float32,
            ),
        },
    }

    res = compute_per_class_pr(dump_data, iou_threshold=0.5)
    laptop_res = res["Laptop"]

    assert laptop_res.num_gt == 2
    assert laptop_res.num_det == 1
    assert np.array_equal(laptop_res.tp, [1.0])
    assert np.array_equal(laptop_res.fp, [0.0])
    assert np.allclose(laptop_res.recalls, [0.5])
    assert np.allclose(laptop_res.precisions, [1.0])
    assert pytest.approx(laptop_res.ap, abs=1e-5) == 0.5


def test_random_false_positive_detector():
    """
    Toy set 3: Random / noisy detector.
    1 ground-truth Chair.
    4 detections: 3 false positives with high scores (0.9, 0.8, 0.7),
    and 1 true positive with low score (0.3).
    Analytical calculation:
      - Ranks 1..3: Precision = 0.0, Recall = 0.0
      - Rank 4: TP=1, FP=3 -> Precision = 1/4 = 0.25, Recall = 1/1 = 1.0
      - Interpolated envelope AP = 1.0 * 0.25 = 0.25.
    """
    dump_data = {
        "detections": {
            "image_ids": np.array(["img_1", "img_1", "img_1", "img_1"]),
            "class_ids": np.array([3, 3, 3, 3], dtype=np.int32),
            "class_names": np.array(["Chair", "Chair", "Chair", "Chair"]),
            "scores": np.array([0.90, 0.80, 0.70, 0.30], dtype=np.float32),
            "boxes": np.array(
                [
                    [0.70, 0.70, 0.90, 0.90],  # FP
                    [0.80, 0.10, 0.95, 0.25],  # FP
                    [0.05, 0.60, 0.20, 0.75],  # FP
                    [0.10, 0.10, 0.40, 0.40],  # TP
                ],
                dtype=np.float32,
            ),
        },
        "ground_truth": {
            "image_ids": np.array(["img_1"]),
            "class_ids": np.array([3], dtype=np.int32),
            "class_names": np.array(["Chair"]),
            "boxes": np.array([[0.10, 0.10, 0.40, 0.40]], dtype=np.float32),
        },
    }

    res = compute_per_class_pr(dump_data, iou_threshold=0.5)
    chair_res = res["Chair"]

    assert chair_res.num_gt == 1
    assert chair_res.num_det == 4
    assert np.array_equal(chair_res.tp, [0.0, 0.0, 0.0, 1.0])
    assert np.array_equal(chair_res.fp, [1.0, 1.0, 1.0, 0.0])
    assert np.allclose(chair_res.precisions, [0.0, 0.0, 0.0, 0.25])
    assert np.allclose(chair_res.recalls, [0.0, 0.0, 0.0, 1.0])
    assert pytest.approx(chair_res.ap, abs=1e-5) == 0.25


def test_duplicate_detections_one_gt():
    """
    Toy set 4: Two detections overlapping the same single GT box.
    First detection (score 0.9) matches (TP).
    Second detection (score 0.8) cannot re-match (FP).
    """
    dump_data = {
        "detections": {
            "image_ids": np.array(["img_1", "img_1"]),
            "class_ids": np.array([1, 1], dtype=np.int32),
            "class_names": np.array(["Person", "Person"]),
            "scores": np.array([0.90, 0.80], dtype=np.float32),
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.50, 0.50],
                    [0.11, 0.11, 0.51, 0.51],
                ],
                dtype=np.float32,
            ),
        },
        "ground_truth": {
            "image_ids": np.array(["img_1"]),
            "class_ids": np.array([1], dtype=np.int32),
            "class_names": np.array(["Person"]),
            "boxes": np.array([[0.10, 0.10, 0.50, 0.50]], dtype=np.float32),
        },
    }

    res = compute_per_class_pr(dump_data, iou_threshold=0.5)
    p_res = res["Person"]

    assert np.array_equal(p_res.tp, [1.0, 0.0])
    assert np.array_equal(p_res.fp, [0.0, 1.0])
    assert np.allclose(p_res.precisions, [1.0, 0.5])
    assert np.allclose(p_res.recalls, [1.0, 1.0])
    assert pytest.approx(p_res.ap, abs=1e-5) == 1.0


def test_calculate_ap_edge_cases():
    """Test calculate_ap on empty and boundary inputs."""
    assert calculate_ap(np.array([]), np.array([])) == 0.0
    assert calculate_ap(np.array([0.5]), np.array([1.0])) == 0.5


def test_compute_per_class_pr_on_fixture():
    """Evaluate against the versioned test fixture from Commit 1."""
    assert FIXTURE_PATH.exists()
    dump_data = load_predictions_dump(FIXTURE_PATH)

    results = compute_per_class_pr(dump_data, iou_threshold=0.5)
    assert len(results) >= 4
    for cname in ["Person", "Laptop", "Chair", "Car"]:
        assert cname in results
        res = results[cname]
        assert res.num_gt > 0
        assert res.num_det > 0
        assert 0.0 <= res.ap <= 1.0

    summary = compute_ap_summary(results)
    assert isinstance(summary, dict)
    assert len(summary) == len(results)

    m_ap = mean_average_precision(results)
    assert 0.0 < m_ap <= 1.0
