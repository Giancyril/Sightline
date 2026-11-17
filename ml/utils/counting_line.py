"""
ml/utils/counting_line.py
-------------------------
Counting line crossing logic for video analytics.
Tracks object centroids crossing a user-defined 2D line segment in normalized
[0, 1] coordinates.
Features:
  - Direction-aware counting ("in" vs "out" based on cross-product orientation)
  - Per-class and aggregate counts
  - Jitter suppression (minimum movement distance + hysteresis cooldown)
  - Edge-case handling for collinear, parallel, or endpoint contacts
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ml.utils.tracker import Track


@dataclass
class Point2D:
    """Normalized 2D coordinate (x, y) where x, y in [0.0, 1.0]."""

    x: float
    y: float

    @classmethod
    def from_val(cls, val: Any) -> Point2D:
        if isinstance(val, Point2D):
            return val
        if isinstance(val, dict):
            return cls(float(val.get("x", 0.0)), float(val.get("y", 0.0)))
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return cls(float(val[0]), float(val[1]))
        raise ValueError(f"Cannot convert {val!r} to Point2D")


@dataclass
class CrossingEvent:
    """Recorded when a track crosses the line."""

    track_id: int
    class_id: int
    class_name: str
    direction: str  # "in" or "out"
    frame_idx: int
    timestamp_s: float
    position: tuple[float, float]  # (x, y) at crossing point


def _ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Signed 2D cross product: (b - a) x (c - a)."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def segments_intersect(
    p1: Point2D,
    p2: Point2D,
    a: Point2D,
    b: Point2D,
) -> tuple[bool, str | None]:
    """
    Test whether segment p1->p2 intersects trajectory segment a->b.

    Returns:
        (intersected, direction) where direction is 'in' or 'out' if intersected.
    """
    d1 = _ccw(p1, p2, a)
    d2 = _ccw(p1, p2, b)
    d3 = _ccw(a, b, p1)
    d4 = _ccw(a, b, p2)

    # Both endpoints must lie on strictly opposite sides
    if (d1 > 1e-7 and d2 < -1e-7) or (d1 < -1e-7 and d2 > 1e-7):
        if (d3 > 1e-7 and d4 < -1e-7) or (d3 < -1e-7 and d4 > 1e-7):
            direction = "in" if d1 > 0 else "out"
            return True, direction

    return False, None


class CountingLine:
    """
    Directional counting line.

    Parameters:
        p1: Start point (x, y) in [0, 1].
        p2: End point (x, y) in [0, 1].
        min_distance: Minimum travel distance between frames to register (jitter filter).
        cooldown_frames: Minimum frames between consecutive crossings by the same track.
    """

    def __init__(
        self,
        p1: Any = (0.0, 0.5),
        p2: Any = (1.0, 0.5),
        min_distance: float = 0.005,
        cooldown_frames: int = 5,
    ) -> None:
        self.p1 = Point2D.from_val(p1)
        self.p2 = Point2D.from_val(p2)
        self.min_distance = min_distance
        self.cooldown_frames = cooldown_frames

        # Crossings state
        self.events: list[CrossingEvent] = []
        self.counts_by_class: dict[str, dict[str, int]] = {}
        self.total_in: int = 0
        self.total_out: int = 0

        # Anti-jitter: track_id -> (last_crossing_frame, last_direction)
        self._last_crossing: dict[int, tuple[int, str]] = {}

    @staticmethod
    def _track_centroid(bbox: tuple[float, float, float, float]) -> Point2D:
        """Calculate normalized (x, y) centroid from (ymin, xmin, ymax, xmax)."""
        ymin, xmin, ymax, xmax = bbox
        return Point2D(x=(xmin + xmax) / 2.0, y=(ymin + ymax) / 2.0)

    def update(
        self,
        tracks: list[Track],
        frame_idx: int = 0,
        timestamp_s: float = 0.0,
    ) -> list[CrossingEvent]:
        """
        Check track movements in this frame for counting line crossings.

        Args:
            tracks: List of active tracks.
            frame_idx: Current video frame index.
            timestamp_s: Current video timestamp in seconds.

        Returns:
            List of new CrossingEvents generated in this frame.
        """
        new_events: list[CrossingEvent] = []

        for track in tracks:
            if len(track.history) < 2:
                continue

            curr_pos = self._track_centroid(track.bbox)
            prev_pos = self._track_centroid(track.history[-2])

            # Jitter check 1: minimum movement distance
            dx = curr_pos.x - prev_pos.x
            dy = curr_pos.y - prev_pos.y
            dist = math.hypot(dx, dy)
            if dist < self.min_distance:
                continue

            # Check segment intersection
            intersected, direction = segments_intersect(
                self.p1, self.p2, prev_pos, curr_pos
            )
            if not intersected or direction is None:
                continue

            # Jitter check 2: cooldown hysteresis
            last = self._last_crossing.get(track.track_id)
            if last is not None:
                last_frame, last_dir = last
                # Ignore if within cooldown window and same or reverse jitter
                if frame_idx - last_frame < self.cooldown_frames:
                    continue

            # Register valid crossing
            self._last_crossing[track.track_id] = (frame_idx, direction)

            event = CrossingEvent(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                direction=direction,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                position=(curr_pos.x, curr_pos.y),
            )
            self.events.append(event)
            new_events.append(event)

            # Update tally
            if direction == "in":
                self.total_in += 1
            else:
                self.total_out += 1

            cname = track.class_name
            if cname not in self.counts_by_class:
                self.counts_by_class[cname] = {"in": 0, "out": 0, "total": 0}
            self.counts_by_class[cname][direction] += 1
            self.counts_by_class[cname]["total"] += 1

        return new_events

    def get_summary(self) -> dict[str, Any]:
        """Return formatted summary dictionary of all line counts."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_crossings": self.total_in + self.total_out,
            "by_class": self.counts_by_class,
            "line": {
                "p1": {"x": self.p1.x, "y": self.p1.y},
                "p2": {"x": self.p2.x, "y": self.p2.y},
            },
        }
