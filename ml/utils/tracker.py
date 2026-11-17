"""
ml/utils/tracker.py
-------------------
IoU-based Multi-Object Tracker (MOT) for video object tracking.
Assigns persistent integer track IDs across frames based on bounding-box
spatial overlap (IoU), with configurable birth/death rules and max-age
for missed frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.utils.bbox import iou


@dataclass
class Track:
    """Represents a tracked object over time."""

    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax)
    score: float
    age: int = 1  # Total frames since birth
    hits: int = 1  # Total frames successfully matched
    missed_frames: int = 0  # Consecutive frames without detection
    is_active: bool = True
    history: list[tuple[float, float, float, float]] = field(default_factory=list)
    color_hex: str = "#0072B2"
    color_rgb: list[int] = field(default_factory=lambda: [0, 114, 178])

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(self.bbox)


@dataclass
class DetectionItem:
    """Standardized detection input for the tracker."""

    bbox: tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax)
    class_id: int
    class_name: str
    score: float
    color_hex: str = "#0072B2"
    color_rgb: list[int] = field(default_factory=lambda: [0, 114, 178])


class IoUTracker:
    """
    Multi-object tracker using greedy Intersection-over-Union (IoU) matching.

    Parameters:
        iou_threshold: Minimum overlap required to match an existing track.
        max_age: Maximum consecutive missed frames before a track is deactivated.
        min_hits: Minimum number of hits before a track is considered confirmed.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id: int = 1
        self.tracks: list[Track] = []

    @property
    def active_tracks(self) -> list[Track]:
        """Return currently active tracks."""
        return [t for t in self.tracks if t.is_active]

    def _normalize_detection(self, det: Any) -> DetectionItem:
        """Convert dict or object into DetectionItem."""
        if isinstance(det, DetectionItem):
            return det
        if isinstance(det, dict):
            box = det.get("bbox") or det.get("box_normalized")
            if isinstance(box, dict):
                bbox = (box["ymin"], box["xmin"], box["ymax"], box["xmax"])
            elif isinstance(box, (list, tuple)):
                bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            else:
                bbox = (0.0, 0.0, 0.0, 0.0)
            return DetectionItem(
                bbox=bbox,
                class_id=int(det.get("class_id", 0)),
                class_name=str(det.get("class_name", "unknown")),
                score=float(det.get("confidence") or det.get("score", 0.0)),
                color_hex=str(det.get("color_hex", "#0072B2")),
                color_rgb=list(det.get("color_rgb", [0, 114, 178])),
            )
        # Attribute-based (e.g. FrameDetection)
        if hasattr(det, "ymin"):
            bbox = (float(det.ymin), float(det.xmin), float(det.ymax), float(det.xmax))
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)
        return DetectionItem(
            bbox=bbox,
            class_id=getattr(det, "class_id", 0),
            class_name=getattr(det, "class_name", "unknown"),
            score=getattr(det, "confidence", 0.0),
            color_hex=getattr(det, "color_hex", "#0072B2"),
            color_rgb=getattr(det, "color_rgb", [0, 114, 178]),
        )

    def update(
        self,
        detections: list[Any],
        frame_idx: int | None = None,
    ) -> list[Track]:
        """
        Update tracker with detections from the current frame.

        Args:
            detections: List of detections (dicts, DetectionItems, or FrameDetections).
            frame_idx: Optional current frame index.

        Returns:
            List of confirmed active tracks in this frame.
        """
        normalized_dets = [self._normalize_detection(d) for d in detections]
        active = self.active_tracks

        # 1. Compute IoU matrix between active tracks and current detections
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        if active and normalized_dets:
            iou_candidates: list[tuple[float, int, int]] = []
            for t_idx, track in enumerate(active):
                box_t = np.array(track.bbox, dtype=np.float32)
                for d_idx, det in enumerate(normalized_dets):
                    box_d = np.array(det.bbox, dtype=np.float32)
                    sim = iou(box_t, box_d)
                    if sim >= self.iou_threshold:
                        iou_candidates.append((sim, t_idx, d_idx))

            # Greedy assignment by highest IoU first
            iou_candidates.sort(key=lambda x: x[0], reverse=True)

            for _sim, t_idx, d_idx in iou_candidates:
                if t_idx in matched_track_indices or d_idx in matched_det_indices:
                    continue
                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)

                # Update existing track
                track = active[t_idx]
                det = normalized_dets[d_idx]
                track.bbox = det.bbox
                track.score = det.score
                track.class_id = det.class_id
                track.class_name = det.class_name
                track.color_hex = det.color_hex
                track.color_rgb = det.color_rgb
                track.hits += 1
                track.missed_frames = 0
                track.age += 1
                track.history.append(det.bbox)

        # 2. Missed tracks (active tracks with no matching detection)
        for t_idx, track in enumerate(active):
            if t_idx not in matched_track_indices:
                track.missed_frames += 1
                track.age += 1
                if track.missed_frames > self.max_age:
                    track.is_active = False

        # 3. New tracks (unmatched detections)
        for d_idx, det in enumerate(normalized_dets):
            if d_idx not in matched_det_indices:
                new_track = Track(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    score=det.score,
                    age=1,
                    hits=1,
                    missed_frames=0,
                    is_active=True,
                    color_hex=det.color_hex,
                    color_rgb=det.color_rgb,
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # Return confirmed active tracks (hits >= min_hits)
        return [t for t in self.active_tracks if t.hits >= self.min_hits]
