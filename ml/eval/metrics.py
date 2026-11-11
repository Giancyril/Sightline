"""
ml/eval/metrics.py
------------------
Precision-recall metrics, multi-threshold COCO mAP@0.5:0.95 calculation,
and detection error categorization (background, class confusion, localization).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.utils.bbox import iou


def compute_precision_recall_curve(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve and AP at a given IoU threshold.

    Returns:
        recalls: 1D array of recall values at each threshold rank.
        precisions: 1D array of precision values at each threshold rank.
        ap: Area under the interpolated PR curve.
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        ap = 0.0 if len(pred_boxes) > 0 else 1.0
        return np.array([0.0]), np.array([0.0 if len(pred_boxes) > 0 else 1.0]), ap
    if len(pred_boxes) == 0:
        return np.array([0.0]), np.array([1.0]), 0.0

    order = np.argsort(pred_scores)[::-1]
    sorted_boxes = [pred_boxes[i] for i in order]

    tp = np.zeros(len(sorted_boxes))
    fp = np.zeros(len(sorted_boxes))
    matched_gt: set[int] = set()

    for p_idx, p_box in enumerate(sorted_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            cur_iou = iou(np.array(p_box, dtype=float), np.array(g_box, dtype=float))
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[p_idx] = 1.0
            matched_gt.add(best_gt_idx)
        else:
            fp[p_idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(n_gt)
    precisions = cum_tp / (cum_tp + cum_fp)

    # 11-point interpolation or full continuous trapezoid/step interpolation
    # Standard VOC/COCO: append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision to the right
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return recalls, precisions, ap


def compute_coco_map_range(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    iou_thresholds: tuple[float, ...] | None = None,
) -> float:
    """
    Compute COCO mAP@[0.5:0.95] by averaging AP across IoU thresholds from 0.50 to 0.95
    with step 0.05 (10 evaluation points).
    """
    if iou_thresholds is None:
        iou_thresholds = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

    aps = [
        compute_precision_recall_curve(pred_boxes, pred_scores, gt_boxes, iou_thresh)[2]
        for iou_thresh in iou_thresholds
    ]
    return float(np.mean(aps))


def categorize_detection_errors(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    pred_classes: list[int],
    gt_boxes: list[list[float]],
    gt_classes: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    """
    Categorize predictions and ground-truth errors following standard diagnostic taxonomy:
      - true_positives (TP): correct class, IoU >= iou_threshold
      - loc_errors: correct class, 0.1 <= IoU < iou_threshold
      - class_confusion: IoU >= iou_threshold with GT of different class
      - background_confusion: max IoU < 0.1 with all GT objects
      - false_negatives: GT objects that were never detected
    """
    counts = {
        "true_positives": 0,
        "loc_errors": 0,
        "class_confusion": 0,
        "background_confusion": 0,
        "false_negatives": 0,
    }

    if len(pred_boxes) == 0:
        counts["false_negatives"] = len(gt_boxes)
        return counts

    order = np.argsort(pred_scores)[::-1]
    matched_gt: set[int] = set()

    for idx in order:
        p_box = pred_boxes[idx]
        p_cls = pred_classes[idx]

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, (g_box, _) in enumerate(zip(gt_boxes, gt_classes, strict=False)):
            cur_iou = iou(np.array(p_box, dtype=float), np.array(g_box, dtype=float))
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            g_cls = gt_classes[best_gt_idx]
            if p_cls == g_cls and best_gt_idx not in matched_gt:
                counts["true_positives"] += 1
                matched_gt.add(best_gt_idx)
            elif p_cls != g_cls:
                counts["class_confusion"] += 1
            else:
                # duplicate detection on already-matched GT
                counts["loc_errors"] += 1
        elif best_gt_idx >= 0 and best_iou >= 0.1:
            g_cls = gt_classes[best_gt_idx]
            if p_cls == g_cls:
                counts["loc_errors"] += 1
            else:
                counts["class_confusion"] += 1
        else:
            counts["background_confusion"] += 1

    counts["false_negatives"] = max(0, len(gt_boxes) - len(matched_gt))
    return counts


def tune_thresholds(
    threshold_candidates: list[float],
    precisions_per_thresh: list[float],
    recalls_per_thresh: list[float],
) -> dict[str, Any]:
    """
    Find optimal confidence threshold balancing precision and recall via F1 score.
    """
    best_f1 = -1.0
    best_thresh = threshold_candidates[0]
    best_prec = 0.0
    best_rec = 0.0

    table = []
    for t, p, r in zip(
        threshold_candidates, precisions_per_thresh, recalls_per_thresh, strict=True
    ):
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        table.append(
            {
                "threshold": t,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_prec = p
            best_rec = r

    return {
        "best_threshold": best_thresh,
        "best_f1": round(best_f1, 4),
        "precision_at_best": round(best_prec, 4),
        "recall_at_best": round(best_rec, 4),
        "sweep_table": table,
    }
