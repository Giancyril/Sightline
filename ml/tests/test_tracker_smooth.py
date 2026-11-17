"""
ml/tests/test_tracker_smooth.py
-------------------------------
Tests for motion model, track smoothing, class consistency,
and occlusion re-entry recovery (Commit 5).
"""

from __future__ import annotations

import numpy as np

from ml.utils.tracker import DetectionItem, IoUTracker, Track


def test_constant_velocity_prediction():
    """Motion model predicts the next bounding box based on velocity."""
    t = Track(
        track_id=1,
        class_id=1,
        class_name="Car",
        bbox=(0.2, 0.2, 0.4, 0.4),
        score=0.9,
        history=[(0.2, 0.1, 0.4, 0.3), (0.2, 0.2, 0.4, 0.4)],  # dx = +0.1
    )
    predicted = t.predict_next_bbox()
    # xmin should be 0.2 + 0.1 = 0.3, xmax should be 0.4 + 0.1 = 0.5
    assert abs(predicted[1] - 0.3) < 1e-4
    assert abs(predicted[3] - 0.5) < 1e-4


def test_class_consistency_prevents_id_swap():
    """
    Two objects of different classes occupying overlapping coordinates
    do NOT swap IDs because class consistency gating is enforced.
    """
    tracker = IoUTracker(iou_threshold=0.2, class_consistency=True)

    # Frame 0: Track 1 = Person, Track 2 = Bicycle separated
    tracker.update(
        [
            {"bbox": (0.3, 0.3, 0.6, 0.5), "class_id": 1, "class_name": "Person"},
            {"bbox": (0.3, 0.55, 0.6, 0.75), "class_id": 2, "class_name": "Bicycle"},
        ]
    )
    assert {t.track_id: t.class_name for t in tracker.active_tracks} == {
        1: "Person",
        2: "Bicycle",
    }

    # Frame 1: Person and Bicycle are near the same coordinate (overlap)
    # Even if Bicycle is presented closest to Person's old box, classes must match!
    active = tracker.update(
        [
            {"bbox": (0.3, 0.35, 0.6, 0.55), "class_id": 2, "class_name": "Bicycle"},
            {"bbox": (0.3, 0.32, 0.6, 0.52), "class_id": 1, "class_name": "Person"},
        ]
    )

    track_classes = {t.track_id: t.class_name for t in active}
    assert track_classes[1] == "Person"
    assert track_classes[2] == "Bicycle"


def test_occlusion_reentry_recovery():
    """
    An object occluded for 8 frames (> max_age=3) enters cemetery,
    then is revived with its original track ID upon reappearing.
    """
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_age=3,
        max_reentry_age=20,
        reentry_iou_threshold=0.2,
    )

    box = (0.2, 0.3, 0.5, 0.6)

    # Frame 0: Object 1 detected
    tracker.update([{"bbox": box, "class_id": 1, "class_name": "Car"}])
    assert tracker.active_tracks[0].track_id == 1

    # Frames 1..8: Object occluded (e.g. driving behind billboard)
    for _ in range(8):
        tracker.update([])

    # Track 1 is now inactive and in cemetery
    assert len(tracker.active_tracks) == 0
    assert any(t.track_id == 1 for t in tracker.cemetery)

    # Frame 9: Car emerges at similar position
    active = tracker.update(
        [{"bbox": (0.2, 0.32, 0.5, 0.62), "class_id": 1, "class_name": "Car"}]
    )
    assert len(active) == 1
    # Revived with original track ID (no ID switch!)
    assert active[0].track_id == 1
    assert active[0].is_active is True
    # Removed from cemetery
    assert not any(t.track_id == 1 for t in tracker.cemetery)


def test_coordinate_smoothing_reduces_jitter():
    """
    Smoothing reduces coordinate standard deviation compared to
    noisy raw observations.
    """
    tracker = IoUTracker(iou_threshold=0.2, smooth_factor=0.5)

    base_ymin = 0.3
    raw_ymins: list[float] = []
    smoothed_ymins: list[float] = []

    for f in range(20):
        # Add alternating artificial jitter
        jitter = 0.04 if f % 2 == 0 else -0.04
        noisy_ymin = base_ymin + jitter
        raw_ymins.append(noisy_ymin)

        active = tracker.update(
            [
                DetectionItem(
                    bbox=(noisy_ymin, 0.2, noisy_ymin + 0.3, 0.5),
                    class_id=1,
                    class_name="Dog",
                    score=0.9,
                )
            ]
        )
        smoothed_ymins.append(active[0].bbox[0])

    # Standard deviation of smoothed coordinates must be strictly less than raw
    assert np.std(smoothed_ymins) < np.std(raw_ymins)


def test_cemetery_permanent_expiry():
    """
    If an object remains missing longer than max_reentry_age, it is permanently
    forgotten and a subsequent detection receives a fresh ID.
    """
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_age=2,
        max_reentry_age=5,
        reentry_iou_threshold=0.2,
    )

    box = (0.2, 0.2, 0.4, 0.4)
    tracker.update([{"bbox": box, "class_id": 1, "class_name": "Chair"}])
    assert tracker.active_tracks[0].track_id == 1

    # Miss 10 frames (> max_reentry_age=5)
    for _ in range(10):
        tracker.update([])

    # Object appears: should get a NEW track ID (2, not 1)
    active = tracker.update([{"bbox": box, "class_id": 1, "class_name": "Chair"}])
    assert len(active) == 1
    assert active[0].track_id == 2
