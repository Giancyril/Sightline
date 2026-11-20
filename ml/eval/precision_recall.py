"""
ml/eval/precision_recall.py
---------------------------
Per-class precision-recall computation and Average Precision (AP) calculation
from validation prediction dumps (Commit 2).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from ml.eval.dump_predictions import load_predictions_dump
from ml.utils.bbox import iou


@dataclass
class ClassPRResult:
    """Precision-recall evaluation results for a single object class."""

    class_id: int
    class_name: str
    scores: np.ndarray  # [N] sorted descending
    tp: np.ndarray  # [N] 1.0 for TP, 0.0 for FP
    fp: np.ndarray  # [N] 1.0 for FP, 0.0 for TP
    precisions: np.ndarray  # [N]
    recalls: np.ndarray  # [N]
    ap: float
    num_gt: int
    num_det: int


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision using all-point interpolated precision envelope.
    (VOC/COCO standard).
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision to the right
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return max(0.0, min(1.0, ap))


def compute_per_class_pr(
    dump_data: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, ClassPRResult]:
    """
    Compute per-class true-positive flags, PR curves, and AP from dump data.

    Matches detections to ground-truth boxes per image and per class:
      - IoU >= iou_threshold (default 0.5)
      - One match per ground truth (first detection by score gets the match)

    Args:
        dump_data: Loaded dump containing 'detections' and 'ground_truth'.
        iou_threshold: Minimum spatial overlap for True Positive.

    Returns:
        Dict mapping class_name -> ClassPRResult.
    """
    dets = dump_data["detections"]
    gts = dump_data["ground_truth"]

    det_img_ids = np.array(dets["image_ids"], dtype=str)
    det_class_names = np.array(dets["class_names"], dtype=str)
    det_class_ids = np.array(dets["class_ids"], dtype=np.int32)
    det_scores = np.array(dets["scores"], dtype=np.float32)
    det_boxes = np.array(dets["boxes"], dtype=np.float32)

    gt_img_ids = np.array(gts["image_ids"], dtype=str)
    gt_class_names = np.array(gts["class_names"], dtype=str)
    gt_class_ids = np.array(gts["class_ids"], dtype=np.int32)
    gt_boxes = np.array(gts["boxes"], dtype=np.float32)

    # Unique classes across both detections and ground truths
    all_classes = sorted(
        set(det_class_names).union(set(gt_class_names)),
        key=lambda c: c.lower(),
    )

    results: dict[str, ClassPRResult] = {}

    for cname in all_classes:
        # Match class_id
        cid_match = det_class_ids[det_class_names == cname]
        cid = int(cid_match[0]) if len(cid_match) > 0 else 0
        if cid == 0:
            gt_cid_match = gt_class_ids[gt_class_names == cname]
            cid = int(gt_cid_match[0]) if len(gt_cid_match) > 0 else 0

        # Filter indices for this class
        det_mask = det_class_names == cname
        gt_mask = gt_class_names == cname

        c_det_img_ids = det_img_ids[det_mask]
        c_det_scores = det_scores[det_mask]
        c_det_boxes = det_boxes[det_mask]

        c_gt_img_ids = gt_img_ids[gt_mask]
        c_gt_boxes = gt_boxes[gt_mask]

        num_gt = len(c_gt_img_ids)
        num_det = len(c_det_img_ids)

        if num_det == 0:
            results[cname] = ClassPRResult(
                class_id=cid,
                class_name=cname,
                scores=np.array([], dtype=np.float32),
                tp=np.array([], dtype=np.float32),
                fp=np.array([], dtype=np.float32),
                precisions=np.array([], dtype=np.float32),
                recalls=np.array([], dtype=np.float32),
                ap=0.0,
                num_gt=num_gt,
                num_det=0,
            )
            continue

        # Per-image greedy matching
        # Find unique images present for this class
        unique_imgs = set(c_det_img_ids).union(set(c_gt_img_ids))
        img_tp_flags: dict[str, list[float]] = {img: [] for img in unique_imgs}
        img_det_indices: dict[str, list[int]] = {img: [] for img in unique_imgs}

        for i, img_id in enumerate(c_det_img_ids):
            img_det_indices[img_id].append(i)

        for img_id in unique_imgs:
            det_idxs = img_det_indices[img_id]
            if not det_idxs:
                continue

            # Detections on this image
            cur_boxes = c_det_boxes[det_idxs]
            cur_scores = c_det_scores[det_idxs]

            # GTs on this image
            gt_idxs = [j for j, gid in enumerate(c_gt_img_ids) if gid == img_id]
            cur_gt_boxes = c_gt_boxes[gt_idxs]

            # Sort image detections descending by score
            order = np.argsort(cur_scores)[::-1]
            matched_gts: set[int] = set()

            tp_local = np.zeros(len(det_idxs), dtype=np.float32)

            for rank_idx in order:
                p_box = cur_boxes[rank_idx]
                best_iou = 0.0
                best_gt = -1

                for g_idx, g_box in enumerate(cur_gt_boxes):
                    if g_idx in matched_gts:
                        continue
                    overlap = iou(p_box, g_box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_gt = g_idx

                if best_iou >= iou_threshold and best_gt >= 0:
                    tp_local[rank_idx] = 1.0
                    matched_gts.add(best_gt)
                else:
                    tp_local[rank_idx] = 0.0

            img_tp_flags[img_id] = list(tp_local)

        # Assemble global per-class arrays
        tp_all = np.zeros(num_det, dtype=np.float32)
        for img_id, det_idxs in img_det_indices.items():
            for local_i, global_i in enumerate(det_idxs):
                tp_all[global_i] = img_tp_flags[img_id][local_i]

        fp_all = 1.0 - tp_all

        # Sort all class detections globally by score descending
        global_order = np.argsort(c_det_scores)[::-1]
        sorted_scores = c_det_scores[global_order]
        sorted_tp = tp_all[global_order]
        sorted_fp = fp_all[global_order]

        cum_tp = np.cumsum(sorted_tp)
        cum_fp = np.cumsum(sorted_fp)

        recalls = (
            cum_tp / float(max(1, num_gt)) if num_gt > 0 else np.zeros_like(cum_tp)
        )
        precisions = cum_tp / (cum_tp + cum_fp)

        ap = calculate_ap(recalls, precisions) if num_gt > 0 else 0.0

        results[cname] = ClassPRResult(
            class_id=cid,
            class_name=cname,
            scores=sorted_scores,
            tp=sorted_tp,
            fp=sorted_fp,
            precisions=precisions,
            recalls=recalls,
            ap=ap,
            num_gt=num_gt,
            num_det=num_det,
        )

    return results


def compute_ap_summary(pr_results: dict[str, ClassPRResult]) -> dict[str, float]:
    """Extract class_name -> AP dictionary from PR results."""
    return {cname: res.ap for cname, res in pr_results.items()}


def mean_average_precision(pr_results: dict[str, ClassPRResult]) -> float:
    """Calculate overall mean Average Precision (mAP) across classes with GT > 0."""
    valid_aps = [res.ap for res in pr_results.values() if res.num_gt > 0]
    return float(np.mean(valid_aps)) if valid_aps else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-class Precision-Recall and AP."
    )
    parser.add_argument(
        "--dump", required=True, help="Path to predictions dump .npz file."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for TP matching.",
    )
    args = parser.parse_args()

    dump_data = load_predictions_dump(args.dump)
    results = compute_per_class_pr(dump_data, iou_threshold=args.iou_threshold)
    m_ap = mean_average_precision(results)

    print("\n" + "=" * 65)
    print(f"  PER-CLASS AVERAGE PRECISION (IoU >= {args.iou_threshold})")
    print("=" * 65)
    print(f"{'Class':<20} {'GT':<6} {'Detections':<12} {'AP@0.5':<10}")
    print("-" * 65)
    for cname, res in sorted(results.items()):
        print(f"{cname:<20} {res.num_gt:<6} {res.num_det:<12} {res.ap * 100:>6.2f}%")
    print("-" * 65)
    print(f"{'Mean AP (mAP)':<20} {'':<6} {'':<12} {m_ap * 100:>6.2f}%\n")


if __name__ == "__main__":
    main()
