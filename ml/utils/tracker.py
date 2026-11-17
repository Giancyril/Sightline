"""
ml/utils/tracker.py
-------------------
IoU-based Multi-Object Tracker (MOT) for video object tracking.
Features:
  - Persistent integer track IDs
  - Track birth and death rules with configurable max_age
  - Constant-velocity motion model for position prediction
  - Exponential moving-average track smoothing to eliminate bounding-box jitter
  - Class-consistency gating to prevent cross-category ID switches
  - Cemetery / re-entry pool to recover tracks after temporary occlusions
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
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    color_hex: str = "#0072B2"
    color_rgb: list[int] = field(default_factory=lambda: [0, 114, 178])

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(self.bbox)

    def predict_next_bbox(self) -> tuple[float, float, float, float]:
        """
        Predict next bounding box using constant-velocity motion model.
        Falls back to last observed bbox if history < 2.
        """
        if len(self.history) < 2:
            return self.bbox

        curr = np.array(self.bbox, dtype=np.float32)
        prev = np.array(self.history[-2], dtype=np.float32)
        v = curr - prev
        pred = curr + v

        # Clip normalized coordinates
        ymin, xmin, ymax, xmax = np.clip(pred, 0.0, 1.0)
        ymax = max(ymin + 0.005, ymax)
        xmax = max(xmin + 0.005, xmax)
        return (float(ymin), float(xmin), float(ymax), float(xmax))


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
    Multi-object tracker using IoU matching, motion prediction,
    class consistency, track smoothing, and occlusion re-entry recovery.

    Parameters:
        iou_threshold: Minimum overlap required to match an active track.
        max_age: Consecutive missed frames before a track is deactivated.
        min_hits: Minimum number of hits before a track is considered confirmed.
        class_consistency: Disallow matches between different class IDs.
        smooth_factor: EMA weight on new detection bbox [0.0..1.0]. (1.0 = no smoothing).
        max_reentry_age: Max missed frames to keep inactive track in cemetery for re-entry.
        reentry_iou_threshold: Minimum overlap to recover a track from cemetery.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 1,
        class_consistency: bool = True,
        smooth_factor: float = 0.8,
        max_reentry_age: int = 30,
        reentry_iou_threshold: float = 0.2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.class_consistency = class_consistency
        self.smooth_factor = smooth_factor
        self.max_reentry_age = max_reentry_age
        self.reentry_iou_threshold = reentry_iou_threshold

        self._next_id: int = 1
        self.tracks: list[Track] = []
        self.cemetery: list[Track] = []

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

    def _smooth_bbox(
        self,
        predicted: tuple[float, float, float, float],
        observed: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Apply exponential moving average to reduce box jitter."""
        alpha = self.smooth_factor
        smoothed = tuple(
            float(alpha * obs + (1.0 - alpha) * pred)
            for pred, obs in zip(predicted, observed, strict=False)
        )
        ymin, xmin, ymax, xmax = np.clip(smoothed, 0.0, 1.0)
        ymax = max(ymin + 0.005, ymax)
        xmax = max(xmin + 0.005, xmax)
        return (float(ymin), float(xmin), float(ymax), float(xmax))

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

        # 1. Match active tracks against detections using motion-predicted locations
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        if active and normalized_dets:
            candidates: list[tuple[float, int, int]] = []
            for t_idx, track in enumerate(active):
                pred_bbox = track.predict_next_bbox()
                box_t = np.array(pred_bbox, dtype=np.float32)

                for d_idx, det in enumerate(normalized_dets):
                    # Class consistency check
                    if self.class_consistency and track.class_id != det.class_id:
                        continue

                    box_d = np.array(det.bbox, dtype=np.float32)
                    sim = iou(box_t, box_d)
                    if sim >= self.iou_threshold:
                        candidates.append((sim, t_idx, d_idx))

            # Greedy assignment by highest IoU first
            candidates.sort(key=lambda x: x[0], reverse=True)

            for _sim, t_idx, d_idx in candidates:
                if t_idx in matched_track_indices or d_idx in matched_det_indices:
                    continue
                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)

                track = active[t_idx]
                det = normalized_dets[d_idx]
                pred_bbox = track.predict_next_bbox()
                new_bbox = self._smooth_bbox(pred_bbox, det.bbox)

                # Velocity update
                prev_bbox = np.array(track.bbox, dtype=np.float32)
                curr_bbox = np.array(new_bbox, dtype=np.float32)
                track.velocity = tuple(float(v) for v in (curr_bbox - prev_bbox))

                track.bbox = new_bbox
                track.score = det.score
                track.class_id = det.class_id
                track.class_name = det.class_name
                track.color_hex = det.color_hex
                track.color_rgb = det.color_rgb
                track.hits += 1
                track.missed_frames = 0
                track.age += 1
                track.history.append(new_bbox)

        # 2. Update missed tracks & move expired ones to cemetery
        for t_idx, track in enumerate(active):
            if t_idx not in matched_track_indices:
                track.missed_frames += 1
                track.age += 1
                if track.missed_frames > self.max_age:
                    track.is_active = False
                    if track not in self.cemetery:
                        self.cemetery.append(track)

        # Age cemetery tracks and purge those exceeding max_reentry_age
        for track in self.cemetery:
            track.missed_frames += 1
            track.age += 1
        self.cemetery = [
            t for t in self.cemetery if t.missed_frames <= self.max_reentry_age
        ]

        # 3. Re-entry check: match remaining detections against cemetery
        unmatched_det_indices = [
            i for i in range(len(normalized_dets)) if i not in matched_det_indices
        ]

        if self.cemetery and unmatched_det_indices:
            cemetery_candidates: list[tuple[float, int, int]] = []
            for c_idx, track in enumerate(self.cemetery):
                if track.missed_frames > self.max_reentry_age:
                    continue
                # For re-entry, use last known position
                box_c = np.array(track.bbox, dtype=np.float32)

                for d_idx in unmatched_det_indices:
                    det = normalized_dets[d_idx]
                    if self.class_consistency and track.class_id != det.class_id:
                        continue
                    box_d = np.array(det.bbox, dtype=np.float32)
                    sim = iou(box_c, box_d)
                    if sim >= self.reentry_iou_threshold:
                        cemetery_candidates.append((sim, c_idx, d_idx))

            cemetery_candidates.sort(key=lambda x: x[0], reverse=True)
            matched_cemetery_indices: set[int] = set()

            for _sim, c_idx, d_idx in cemetery_candidates:
                if c_idx in matched_cemetery_indices or d_idx in matched_det_indices:
                    continue
                matched_cemetery_indices.add(c_idx)
                matched_det_indices.add(d_idx)

                # Revive track from cemetery!
                track = self.cemetery[c_idx]
                det = normalized_dets[d_idx]
                track.is_active = True
                track.missed_frames = 0
                track.bbox = det.bbox
                track.score = det.score
                track.hits += 1
                track.age += 1
                track.history.append(det.bbox)

            # Prune revived tracks from cemetery
            self.cemetery = [
                t
                for idx, t in enumerate(self.cemetery)
                if idx not in matched_cemetery_indices
                and t.missed_frames <= self.max_reentry_age
            ]

        # 4. New track birth for any remaining unmatched detections
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
