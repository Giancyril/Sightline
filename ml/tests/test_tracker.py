"""
ml/tests/test_tracker.py
------------------------
Unit tests for IoU-based multi-object tracker (Commit 4).
Covers:
  - Persistent IDs over time
  - Track birth and death rules
  - Max-age for missed frames
  - Synthetic sequence for ID stability
  - Crossing paths scenario (swapping prevention)
  - Detection input normalization
"""

from __future__ import annotations

from ml.utils.tracker import DetectionItem, IoUTracker


def test_track_birth_single_object():
    """First detection in a video creates track_id=1 with active status."""
    tracker = IoUTracker(iou_threshold=0.3, max_age=3)
    dets = [
        {
            "class_id": 1,
            "class_name": "Person",
            "bbox": (0.1, 0.1, 0.4, 0.3),
            "confidence": 0.85,
        }
    ]
    tracks = tracker.update(dets, frame_idx=0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_name == "Person"
    assert tracks[0].hits == 1
    assert tracks[0].age == 1
    assert tracks[0].is_active is True


def test_id_persistence_stationary_and_moving():
    """An object moving gradually retains its unique track ID across 10 frames."""
    tracker = IoUTracker(iou_threshold=0.3, max_age=3)

    # Object moving rightwards by 0.01 per frame
    for f in range(10):
        xmin = 0.1 + f * 0.01
        xmax = xmin + 0.2
        dets = [
            DetectionItem(
                bbox=(0.2, xmin, 0.5, xmax),
                class_id=1,
                class_name="Car",
                score=0.9,
            )
        ]
        active = tracker.update(dets, frame_idx=f)
        assert len(active) == 1
        assert active[0].track_id == 1
        assert active[0].hits == f + 1
        assert active[0].age == f + 1
        assert active[0].missed_frames == 0


def test_track_death_after_max_age():
    """A track that goes undetected for more than max_age frames is marked inactive."""
    max_age = 3
    tracker = IoUTracker(iou_threshold=0.3, max_age=max_age)

    # Frame 0: object appears
    tracker.update([{"bbox": (0.1, 0.1, 0.3, 0.3), "class_id": 1, "class_name": "Dog"}])
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert track.track_id == 1

    # Frames 1..3: object missing (missed_frames = 1, 2, 3 <= max_age)
    for _ in range(max_age):
        tracker.update([])
        assert track.is_active is True

    # Frame 4: missed_frames = 4 > max_age -> deactivated
    tracker.update([])
    assert track.is_active is False
    assert len(tracker.active_tracks) == 0


def test_track_recovery_before_max_age():
    """If an object is missed for 2 frames (< max_age=3) and reappears, it retains its ID."""
    tracker = IoUTracker(iou_threshold=0.3, max_age=3)
    box = (0.2, 0.2, 0.5, 0.5)

    # Frame 0: visible
    tracker.update([{"bbox": box, "class_id": 1, "class_name": "Bus"}])
    assert tracker.active_tracks[0].track_id == 1

    # Frames 1, 2: temporarily occluded / missed
    tracker.update([])
    tracker.update([])
    assert tracker.tracks[0].missed_frames == 2
    assert tracker.tracks[0].is_active is True

    # Frame 3: reappears in same spot
    active = tracker.update([{"bbox": box, "class_id": 1, "class_name": "Bus"}])
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].missed_frames == 0
    assert active[0].hits == 2


def test_crossing_paths_id_stability():
    """
    Two objects crossing paths retain their respective IDs without swapping.
    Object A moves left-to-right from x=0.1 to x=0.8.
    Object B moves right-to-left from x=0.8 to x=0.1.
    """
    tracker = IoUTracker(iou_threshold=0.2, max_age=3)
    n_frames = 15

    track_a_xs: list[float] = []
    track_b_xs: list[float] = []

    for f in range(n_frames):
        alpha = f / (n_frames - 1)
        # A moves left to right
        x_a = 0.1 + alpha * 0.7
        # B moves right to left
        x_b = 0.8 - alpha * 0.7

        box_a = (0.3, x_a, 0.5, x_a + 0.15)
        box_b = (0.3, x_b, 0.5, x_b + 0.15)

        # Present detections in alternating order to test order independence
        dets = [
            {"bbox": box_a, "class_id": 1, "class_name": "Person", "confidence": 0.9},
            {"bbox": box_b, "class_id": 2, "class_name": "Bicycle", "confidence": 0.85},
        ]
        if f % 2 == 1:
            dets.reverse()

        active = tracker.update(dets, frame_idx=f)
        assert len(active) == 2

        track_by_id = {t.track_id: t for t in active}
        assert 1 in track_by_id and 2 in track_by_id

        track_a_xs.append(track_by_id[1].bbox[1])
        track_by_id[2].bbox[1]
        track_b_xs.append(track_by_id[2].bbox[1])

    # Verify track 1 consistently progressed left-to-right (x increases)
    assert track_a_xs[-1] > track_a_xs[0]
    # Verify track 2 consistently progressed right-to-left (x decreases)
    assert track_b_xs[-1] < track_b_xs[0]


def test_multiple_independent_objects():
    """Tracker correctly tracks 3 distinct objects simultaneously."""
    tracker = IoUTracker(iou_threshold=0.3, max_age=2)

    dets_f0 = [
        {"bbox": (0.1, 0.1, 0.3, 0.3), "class_id": 1, "class_name": "Cat"},
        {"bbox": (0.6, 0.6, 0.8, 0.8), "class_id": 2, "class_name": "Dog"},
    ]
    active = tracker.update(dets_f0)
    assert len(active) == 2
    assert {t.track_id for t in active} == {1, 2}

    # Third object appears in frame 1
    dets_f1 = [
        {"bbox": (0.11, 0.11, 0.31, 0.31), "class_id": 1, "class_name": "Cat"},
        {"bbox": (0.61, 0.61, 0.81, 0.81), "class_id": 2, "class_name": "Dog"},
        {"bbox": (0.1, 0.7, 0.3, 0.9), "class_id": 3, "class_name": "Bird"},
    ]
    active = tracker.update(dets_f1)
    assert len(active) == 3
    assert {t.track_id for t in active} == {1, 2, 3}


def test_detection_dict_format_variants():
    """Tracker handles nested box_normalized dict formats as well as flat bboxes."""
    tracker = IoUTracker(iou_threshold=0.3)
    dets = [
        {
            "class_id": 5,
            "class_name": "Bottle",
            "box_normalized": {"ymin": 0.2, "xmin": 0.3, "ymax": 0.7, "xmax": 0.5},
            "confidence": 0.92,
        }
    ]
    active = tracker.update(dets)
    assert len(active) == 1
    assert active[0].bbox == (0.2, 0.3, 0.7, 0.5)
    assert active[0].class_name == "Bottle"
