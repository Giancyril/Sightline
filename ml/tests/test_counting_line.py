"""
ml/tests/test_counting_line.py
------------------------------
Unit tests for counting line crossing logic (Commit 6).
Covers:
  - Direction-aware crossing ("in" and "out")
  - Vertical and horizontal lines
  - Per-class counts and totals
  - Micro-movement jitter suppression
  - Hysteresis / cooldown jitter suppression
  - Parallel trajectories and stationary tracks
  - Summary dictionary structure
"""

from __future__ import annotations

from ml.utils.counting_line import CountingLine, Point2D, segments_intersect
from ml.utils.tracker import Track


def _make_track(
    track_id: int,
    class_name: str,
    class_id: int,
    history_boxes: list[tuple[float, float, float, float]],
) -> Track:
    """Helper to build a track with explicit position history."""
    return Track(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        bbox=history_boxes[-1],
        score=0.9,
        history=history_boxes,
    )


# ---------------------------------------------------------------------------
# Segments intersection geometric tests
# ---------------------------------------------------------------------------


def test_segment_intersect_direct():
    line_start = Point2D(0.0, 0.5)
    line_end = Point2D(1.0, 0.5)
    traj_start = Point2D(0.5, 0.3)
    traj_end = Point2D(0.5, 0.7)

    intersected, direction = segments_intersect(
        line_start, line_end, traj_start, traj_end
    )
    assert intersected is True
    assert direction in ("in", "out")


def test_segment_parallel_no_intersect():
    line_start = Point2D(0.0, 0.5)
    line_end = Point2D(1.0, 0.5)
    traj_start = Point2D(0.0, 0.3)
    traj_end = Point2D(1.0, 0.3)

    intersected, direction = segments_intersect(
        line_start, line_end, traj_start, traj_end
    )
    assert intersected is False
    assert direction is None


# ---------------------------------------------------------------------------
# Counting line direction-aware tests
# ---------------------------------------------------------------------------


def test_horizontal_line_both_directions():
    """Horizontal line at y=0.5 detects 'in' and 'out' crossings."""
    line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5))

    # Track 1: moves downwards (ymin 0.3->0.6)
    t1 = _make_track(
        track_id=1,
        class_name="Car",
        class_id=1,
        history_boxes=[(0.2, 0.4, 0.4, 0.6), (0.5, 0.4, 0.7, 0.6)],
    )

    # Track 2: moves upwards (ymin 0.6->0.3)
    t2 = _make_track(
        track_id=2,
        class_name="Car",
        class_id=1,
        history_boxes=[(0.5, 0.4, 0.7, 0.6), (0.2, 0.4, 0.4, 0.6)],
    )

    events_1 = line.update([t1], frame_idx=1, timestamp_s=0.1)
    events_2 = line.update([t2], frame_idx=2, timestamp_s=0.2)

    assert len(events_1) == 1
    assert len(events_2) == 1
    # One must be 'in' and the other 'out'
    assert events_1[0].direction != events_2[0].direction
    assert line.total_in == 1
    assert line.total_out == 1


def test_vertical_line_crossings():
    """Vertical line at x=0.5 detects crossings across x."""
    line = CountingLine(p1=(0.5, 0.0), p2=(0.5, 1.0))

    # Track moves from x=0.3 to x=0.7
    t = _make_track(
        track_id=10,
        class_name="Person",
        class_id=2,
        history_boxes=[(0.4, 0.2, 0.6, 0.4), (0.4, 0.6, 0.6, 0.8)],
    )
    events = line.update([t], frame_idx=1, timestamp_s=0.04)
    assert len(events) == 1
    assert events[0].track_id == 10
    assert line.total_in + line.total_out == 1


# ---------------------------------------------------------------------------
# Per-class counts
# ---------------------------------------------------------------------------


def test_per_class_and_total_counts():
    """Counts are grouped accurately by class and directions."""
    line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5))

    car = _make_track(1, "Car", 1, [(0.2, 0.2, 0.4, 0.4), (0.5, 0.2, 0.7, 0.4)])
    bus = _make_track(2, "Bus", 2, [(0.2, 0.6, 0.4, 0.8), (0.5, 0.6, 0.7, 0.8)])

    events = line.update([car, bus], frame_idx=1)
    assert len(events) == 2

    summary = line.get_summary()
    assert summary["total_crossings"] == 2
    assert "Car" in summary["by_class"]
    assert "Bus" in summary["by_class"]
    assert summary["by_class"]["Car"]["total"] == 1
    assert summary["by_class"]["Bus"]["total"] == 1


# ---------------------------------------------------------------------------
# Jitter suppression
# ---------------------------------------------------------------------------


def test_jitter_micro_movements_ignored():
    """Centroid movement smaller than min_distance does not register."""
    line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5), min_distance=0.05)

    # Box moves by only 0.001 across the line
    t = _make_track(
        1,
        "Person",
        1,
        [(0.3995, 0.4, 0.5995, 0.6), (0.4005, 0.4, 0.6005, 0.6)],
    )
    events = line.update([t], frame_idx=1)
    assert len(events) == 0
    assert line.total_in + line.total_out == 0


def test_jitter_cooldown_hysteresis():
    """An object lingering and oscillating on the line triggers only once within cooldown window."""
    line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5), cooldown_frames=10)

    # Cross 1: at frame 1
    t1 = _make_track(1, "Car", 1, [(0.3, 0.4, 0.45, 0.6), (0.55, 0.4, 0.7, 0.6)])
    events_f1 = line.update([t1], frame_idx=1)
    assert len(events_f1) == 1

    # Cross 2 immediately at frame 2 (within cooldown window of 10 frames)
    t2 = _make_track(1, "Car", 1, [(0.55, 0.4, 0.7, 0.6), (0.3, 0.4, 0.45, 0.6)])
    events_f2 = line.update([t2], frame_idx=2)
    # Jitter suppressed by cooldown
    assert len(events_f2) == 0

    # Cross 3 after cooldown has expired (frame 15 > 1 + 10)
    t3 = _make_track(1, "Car", 1, [(0.3, 0.4, 0.45, 0.6), (0.55, 0.4, 0.7, 0.6)])
    events_f15 = line.update([t3], frame_idx=15)
    assert len(events_f15) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_parallel_path_never_crosses():
    """An object moving parallel to the line generates 0 events."""
    line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5))
    t = _make_track(
        1,
        "Bicycle",
        3,
        [(0.2, 0.1, 0.3, 0.2), (0.2, 0.7, 0.3, 0.8)],
    )
    events = line.update([t], frame_idx=1)
    assert len(events) == 0


def test_stationary_track_never_crosses():
    """A stationary object generates 0 events."""
    line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5))
    box = (0.4, 0.4, 0.6, 0.6)
    t = _make_track(1, "Bench", 4, [box, box])
    events = line.update([t], frame_idx=1)
    assert len(events) == 0


def test_summary_structure():
    """Verify get_summary returns correctly formatted structure."""
    line = CountingLine(p1=(0.1, 0.2), p2=(0.8, 0.9))
    summary = line.get_summary()
    assert summary["total_in"] == 0
    assert summary["total_out"] == 0
    assert summary["total_crossings"] == 0
    assert summary["by_class"] == {}
    assert summary["line"]["p1"]["x"] == 0.1
    assert summary["line"]["p2"]["x"] == 0.8
