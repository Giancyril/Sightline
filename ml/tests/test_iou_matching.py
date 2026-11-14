"""
ml/tests/test_iou_matching.py
-----------------------------
Unit tests for pure IoU matching functions:
  - compute_iou accuracy and edge cases
  - match_detections with matched, only-A, only-B
  - class-aware vs class-agnostic matching
  - configurable IoU threshold
  - greedy assignment with multiple overlapping boxes
  - Pydantic schema / dict / tuple compatibility
"""

from __future__ import annotations

import pytest

from api.schemas import DetectionItem, NormalizedBBox, PixelBBox
from ml.utils.iou_matching import MatchResult, compute_iou, match_detections


def test_compute_iou_identical_boxes():
    """Identical bounding boxes should have an IoU of exactly 1.0."""
    box = [0.1, 0.1, 0.5, 0.5]
    assert compute_iou(box, box) == pytest.approx(1.0)


def test_compute_iou_disjoint_boxes():
    """Completely disjoint bounding boxes should have an IoU of 0.0."""
    box_a = [0.0, 0.0, 0.2, 0.2]
    box_b = [0.5, 0.5, 0.8, 0.8]
    assert compute_iou(box_a, box_b) == 0.0


def test_compute_iou_half_overlap():
    """Known partial overlap calculation."""
    # Box A: area = 1.0 * 2.0 = 2.0 (ymin=0, xmin=0, ymax=1, xmax=2)
    # Box B: area = 1.0 * 2.0 = 2.0 (ymin=0, xmin=1, ymax=1, xmax=3)
    # Intersection: y in [0, 1], x in [1, 2] -> area = 1.0
    # Union = 2 + 2 - 1 = 3.0 -> IoU = 1/3
    box_a = [0.0, 0.0, 1.0, 2.0]
    box_b = [0.0, 1.0, 1.0, 3.0]
    assert compute_iou(box_a, box_b) == pytest.approx(1.0 / 3.0)


def test_match_empty_inputs():
    """Empty detection lists should return empty MatchResult."""
    res = match_detections([], [])
    assert isinstance(res, MatchResult)
    assert res.matched == []
    assert res.only_a == []
    assert res.only_b == []

    # Only A present
    det_a = {"box_normalized": [0.1, 0.1, 0.4, 0.4], "class_name": "Dog"}
    res_a = match_detections([det_a], [])
    assert res_a.matched == []
    assert res_a.only_a == [det_a]
    assert res_a.only_b == []

    # Only B present
    det_b = {"box_normalized": [0.1, 0.1, 0.4, 0.4], "class_name": "Dog"}
    res_b = match_detections([], [det_b])
    assert res_b.matched == []
    assert res_b.only_a == []
    assert res_b.only_b == [det_b]


def test_match_single_matched_pair():
    """Single matching box in both models should yield 1 matched, 0 only-A, 0 only-B."""
    det_a = {
        "box_normalized": [0.1, 0.1, 0.5, 0.5],
        "class_name": "Car",
        "confidence": 0.85,
    }
    det_b = {
        "box_normalized": [0.1, 0.1, 0.5, 0.5],
        "class_name": "Car",
        "confidence": 0.90,
    }

    matched, only_a, only_b = match_detections([det_a], [det_b], iou_threshold=0.5)
    assert len(matched) == 1
    assert matched[0][0] == det_a
    assert matched[0][1] == det_b
    assert matched[0][2] == pytest.approx(1.0)
    assert only_a == []
    assert only_b == []


def test_match_disjoint_only_a_and_only_b():
    """Disjoint detections yield 0 matched, 1 only-A, 1 only-B."""
    det_a = {"box_normalized": [0.0, 0.0, 0.2, 0.2], "class_name": "Person"}
    det_b = {"box_normalized": [0.7, 0.7, 0.9, 0.9], "class_name": "Person"}

    res = match_detections([det_a], [det_b], iou_threshold=0.5)
    assert len(res.matched) == 0
    assert res.only_a == [det_a]
    assert res.only_b == [det_b]


def test_class_aware_vs_agnostic():
    """Class mismatch prevents matching under class_aware=True, but allows it under False."""
    det_a = {"box_normalized": [0.2, 0.2, 0.6, 0.6], "class_name": "Cat"}
    det_b = {"box_normalized": [0.2, 0.2, 0.6, 0.6], "class_name": "Dog"}

    # Class-aware: mismatch
    res_aware = match_detections([det_a], [det_b], class_aware=True)
    assert len(res_aware.matched) == 0
    assert res_aware.only_a == [det_a]
    assert res_aware.only_b == [det_b]

    # Class-agnostic: match by spatial overlap
    res_agnostic = match_detections([det_a], [det_b], class_aware=False)
    assert len(res_agnostic.matched) == 1
    assert res_agnostic.only_a == []
    assert res_agnostic.only_b == []


def test_configurable_iou_threshold():
    """Matching respects configurable IoU threshold boundary."""
    # Box A: [0, 0, 1, 2], Box B: [0, 1, 1, 3] -> IoU ~ 0.333
    det_a = {"box_normalized": [0.0, 0.0, 1.0, 2.0], "class_name": "Bus"}
    det_b = {"box_normalized": [0.0, 1.0, 1.0, 3.0], "class_name": "Bus"}

    # Threshold 0.5: should NOT match
    res_high = match_detections([det_a], [det_b], iou_threshold=0.5)
    assert len(res_high.matched) == 0
    assert len(res_high.only_a) == 1
    assert len(res_high.only_b) == 1

    # Threshold 0.3: should match
    res_low = match_detections([det_a], [det_b], iou_threshold=0.3)
    assert len(res_low.matched) == 1
    assert len(res_low.only_a) == 0
    assert len(res_low.only_b) == 0


def test_greedy_highest_iou_assignment():
    """When multiple boxes in B overlap with box A, the one with highest IoU matches."""
    det_a = {"box_normalized": [0.1, 0.1, 0.5, 0.5], "class_name": "Bird"}
    # det_b_close has IoU ~ 0.8
    det_b_close = {
        "box_normalized": [0.12, 0.12, 0.5, 0.5],
        "class_name": "Bird",
        "id": "close",
    }
    # det_b_loose has IoU ~ 0.4
    det_b_loose = {
        "box_normalized": [0.2, 0.2, 0.6, 0.6],
        "class_name": "Bird",
        "id": "loose",
    }

    res = match_detections(
        [det_a], [det_b_close, det_b_loose], iou_threshold=0.3, class_aware=True
    )
    assert len(res.matched) == 1
    assert res.matched[0][0] == det_a
    assert res.matched[0][1]["id"] == "close"
    assert len(res.only_a) == 0
    assert len(res.only_b) == 1
    assert res.only_b[0]["id"] == "loose"


def test_pydantic_detection_item_compatibility():
    """Verify match_detections works directly with Pydantic DetectionItem objects."""
    item_a = DetectionItem(
        class_id=1,
        class_name="person",
        confidence=0.92,
        box_normalized=NormalizedBBox(ymin=0.1, xmin=0.1, ymax=0.8, xmax=0.5),
        box_pixel=PixelBBox(ymin=10, xmin=10, ymax=80, xmax=50, width=40, height=70),
        color_hex="#ff0000",
        color_rgb=[255, 0, 0],
    )
    item_b = DetectionItem(
        class_id=1,
        class_name="person",
        confidence=0.88,
        box_normalized=NormalizedBBox(ymin=0.11, xmin=0.1, ymax=0.8, xmax=0.51),
        box_pixel=PixelBBox(ymin=11, xmin=10, ymax=80, xmax=51, width=41, height=69),
        color_hex="#00ff00",
        color_rgb=[0, 255, 0],
    )

    res = match_detections([item_a], [item_b], iou_threshold=0.5)
    assert len(res.matched) == 1
    assert res.matched[0][0] == item_a
    assert res.matched[0][1] == item_b
    assert res.matched[0][2] > 0.8
    assert res.only_a == []
    assert res.only_b == []
