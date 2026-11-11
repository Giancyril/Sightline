"""
ml/utils/bbox.py
Bounding-box utility functions used across the pipeline.
All boxes are represented as [y_min, x_min, y_max, x_max] in normalized [0,1] coords
unless noted otherwise (pixel-space functions are labelled _px).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def iou(box_a: NDArray[np.float32], box_b: NDArray[np.float32]) -> float:
    """Compute Intersection-over-Union between two boxes [y1, x1, y2, x2]."""
    ya1, xa1, ya2, xa2 = box_a
    yb1, xb1, yb2, xb2 = box_b

    inter_y1 = max(ya1, yb1)
    inter_x1 = max(xa1, xb1)
    inter_y2 = min(ya2, yb2)
    inter_x2 = min(xa2, xb2)

    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    intersection = inter_h * inter_w

    area_a = (ya2 - ya1) * (xa2 - xa1)
    area_b = (yb2 - yb1) * (xb2 - xb1)
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def clip_boxes(
    boxes: NDArray[np.float32],
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> NDArray[np.float32]:
    """Clip normalized boxes to [min_val, max_val]."""
    return np.clip(boxes, min_val, max_val)


def box_area(box: NDArray[np.float32]) -> float:
    """Return area of a single box [y1, x1, y2, x2]."""
    h = max(0.0, float(box[2] - box[0]))
    w = max(0.0, float(box[3] - box[1]))
    return h * w


def nms(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float = 0.5,
) -> list[int]:
    """
    Non-maximum suppression.

    Args:
        boxes: (N, 4) float32 array of [y1, x1, y2, x2] boxes.
        scores: (N,) float32 confidence scores.
        iou_threshold: IoU threshold above which a box is suppressed.

    Returns:
        List of indices of kept boxes in descending score order.
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    kept: list[int] = []

    while order.size > 0:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_threshold]

    return kept
