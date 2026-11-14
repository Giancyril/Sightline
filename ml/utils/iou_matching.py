"""
ml/utils/iou_matching.py
------------------------
Pure IoU matching algorithm to compare detection sets from two models.
Finds matched object pairs, detections unique to model A (only-A),
and detections unique to model B (only-B). Supports configurable IoU
thresholds and class-aware or class-agnostic matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple


class MatchResult(NamedTuple):
    """Result of bipartite detection matching between two models."""

    matched: list[tuple[Any, Any, float]]
    only_a: list[Any]
    only_b: list[Any]


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Compute Intersection-over-Union between two boxes [y1, x1, y2, x2].

    Args:
        box_a: Coordinates [ymin, xmin, ymax, xmax].
        box_b: Coordinates [ymin, xmin, ymax, xmax].

    Returns:
        IoU value in [0.0, 1.0].
    """
    ya1, xa1, ya2, xa2 = box_a
    yb1, xb1, yb2, xb2 = box_b

    inter_y1 = max(ya1, yb1)
    inter_x1 = max(xa1, xb1)
    inter_y2 = min(ya2, yb2)
    inter_x2 = min(xa2, xb2)

    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    intersection = inter_h * inter_w

    area_a = max(0.0, (ya2 - ya1) * (xa2 - xa1))
    area_b = max(0.0, (yb2 - yb1) * (xb2 - xb1))
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0.0 else 0.0


def _extract_box(det: Any) -> list[float]:
    """Extract [ymin, xmin, ymax, xmax] coordinates from various detection representations."""
    if hasattr(det, "box_normalized"):
        b = det.box_normalized
        if hasattr(b, "ymin"):
            return [float(b.ymin), float(b.xmin), float(b.ymax), float(b.xmax)]
        if isinstance(b, (list, tuple)):
            return [float(x) for x in b]

    if isinstance(det, dict):
        if "box_normalized" in det:
            b = det["box_normalized"]
            if isinstance(b, dict):
                return [
                    float(b["ymin"]),
                    float(b["xmin"]),
                    float(b["ymax"]),
                    float(b["xmax"]),
                ]
            if isinstance(b, (list, tuple)):
                return [float(x) for x in b]
        if "box" in det:
            return [float(x) for x in det["box"]]
        if "bbox" in det:
            return [float(x) for x in det["bbox"]]

    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return [float(det[0]), float(det[1]), float(det[2]), float(det[3])]

    raise ValueError(
        f"Unable to extract bounding box coordinates from detection: {det}"
    )


def _extract_class(det: Any) -> str | int | None:
    """Extract class label/identifier from detection representation."""
    if hasattr(det, "class_name"):
        return str(det.class_name).lower()
    if hasattr(det, "class_id"):
        return det.class_id

    if isinstance(det, dict):
        if "class_name" in det:
            return str(det["class_name"]).lower()
        if "class_id" in det:
            return det["class_id"]
        if "label" in det:
            return str(det["label"]).lower()

    return None


def match_detections(
    detections_a: list[Any],
    detections_b: list[Any],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> MatchResult:
    """
    Match object detections from two models using greedy bipartite matching by IoU.

    Args:
        detections_a: List of detections from Model A.
        detections_b: List of detections from Model B.
        iou_threshold: Minimum IoU overlap required to consider two detections a match.
        class_aware: When True, objects only match if their class identifiers match.

    Returns:
        MatchResult namedtuple containing:
          - matched: list of (det_a, det_b, iou_score) tuples
          - only_a: detections from Model A with no corresponding match in B
          - only_b: detections from Model B with no corresponding match in A
    """
    if not detections_a and not detections_b:
        return MatchResult(matched=[], only_a=[], only_b=[])
    if not detections_a:
        return MatchResult(matched=[], only_a=[], only_b=list(detections_b))
    if not detections_b:
        return MatchResult(matched=[], only_a=list(detections_a), only_b=[])

    boxes_a = [_extract_box(d) for d in detections_a]
    boxes_b = [_extract_box(d) for d in detections_b]
    classes_a = [_extract_class(d) for d in detections_a]
    classes_b = [_extract_class(d) for d in detections_b]

    # Compute pairwise candidate matches
    candidates: list[tuple[float, int, int]] = []
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            if class_aware and classes_a[i] != classes_b[j]:
                continue
            overlap = compute_iou(box_a, box_b)
            if overlap >= iou_threshold:
                candidates.append((overlap, i, j))

    # Sort candidates greedily by descending IoU
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_a_indices: set[int] = set()
    matched_b_indices: set[int] = set()
    matched: list[tuple[Any, Any, float]] = []

    for overlap, i, j in candidates:
        if i in matched_a_indices or j in matched_b_indices:
            continue
        matched_a_indices.add(i)
        matched_b_indices.add(j)
        matched.append((detections_a[i], detections_b[j], round(overlap, 4)))

    only_a = [d for i, d in enumerate(detections_a) if i not in matched_a_indices]
    only_b = [d for j, d in enumerate(detections_b) if j not in matched_b_indices]

    return MatchResult(matched=matched, only_a=only_a, only_b=only_b)
