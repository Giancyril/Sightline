"""
ml/tests/test_bbox.py
Unit tests for bounding-box utilities (IoU, NMS, clip, area).
Marked @pytest.mark.stage0 so the Stage-0 gate runs them.
"""

import numpy as np
import pytest

from ml.utils.bbox import box_area, clip_boxes, iou, nms


@pytest.mark.stage0
class TestIoU:
    def test_perfect_overlap(self) -> None:
        box = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        b = np.array([0.6, 0.6, 1.0, 1.0], dtype=np.float32)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        b = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float32)
        # intersection = 0.5*0.5 = 0.25; union = 1 + 1 - 0.25 = 1.75
        assert iou(a, b) == pytest.approx(0.25 / 1.75, rel=1e-5)

    def test_touching_edges(self) -> None:
        a = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        b = np.array([0.5, 0.0, 1.0, 1.0], dtype=np.float32)
        assert iou(a, b) == pytest.approx(0.0)


@pytest.mark.stage0
class TestNMS:
    def test_single_box(self) -> None:
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        assert nms(boxes, scores) == [0]

    def test_suppresses_duplicate(self) -> None:
        boxes = np.array(
            [[0.0, 0.0, 1.0, 1.0], [0.05, 0.05, 0.95, 0.95]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        kept = nms(boxes, scores, iou_threshold=0.5)
        assert kept == [0]

    def test_keeps_non_overlapping(self) -> None:
        boxes = np.array([[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        kept = nms(boxes, scores, iou_threshold=0.5)
        assert set(kept) == {0, 1}

    def test_empty_input(self) -> None:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        assert nms(boxes, scores) == []


@pytest.mark.stage0
class TestClipBoxes:
    def test_clips_above_one(self) -> None:
        boxes = np.array([[0.0, 0.0, 1.5, 1.5]], dtype=np.float32)
        clipped = clip_boxes(boxes)
        assert clipped[0, 2] == pytest.approx(1.0)
        assert clipped[0, 3] == pytest.approx(1.0)

    def test_clips_below_zero(self) -> None:
        boxes = np.array([[-0.1, -0.2, 0.5, 0.5]], dtype=np.float32)
        clipped = clip_boxes(boxes)
        assert clipped[0, 0] == pytest.approx(0.0)
        assert clipped[0, 1] == pytest.approx(0.0)


@pytest.mark.stage0
class TestBoxArea:
    def test_unit_box(self) -> None:
        box = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        assert box_area(box) == pytest.approx(1.0)

    def test_zero_area(self) -> None:
        box = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        assert box_area(box) == pytest.approx(0.0)

    def test_degenerate_box(self) -> None:
        # inverted box — area should be 0
        box = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        assert box_area(box) == pytest.approx(0.0)
