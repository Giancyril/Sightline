"""
ml/eval/active_learning.py
--------------------------
Active learning uncertainty ranking for candidate review images (Day 4 Commit 8).

Calculates an uncertainty score based on:
  1. Detections near decision thresholds (borderline confidence, using Day 3 per-class thresholds).
  2. Model disagreement (unmatched bounding boxes between two models via Day 1 IoU bipartite matching).

Returns structured uncertainty score and human-readable explanation reasons for each ranked image.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.data.classes import CLASS_BY_ID
from ml.utils.iou_matching import match_detections


@dataclass
class ImageUncertaintyInput:
    """Input payload for scoring uncertainty of an image's detections."""

    image_id: str
    detections_primary: list[Any]
    detections_secondary: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedImage:
    """Uncertainty evaluation output and ranking for a candidate image."""

    image_id: str
    score: float
    rank: int
    reasons: list[str]
    threshold_uncertainty_score: float
    disagreement_score: float
    num_near_threshold: int
    num_disagreements: int
    metadata: dict[str, Any] = field(default_factory=dict)


def load_default_thresholds(
    thresholds_path: str | Path | None = None,
) -> tuple[dict[str, float], float]:
    """Load per-class recommended thresholds from thresholds.json, with global fallback."""
    if thresholds_path is None:
        thresholds_path = Path("models/fine-tuned/thresholds.json")
    else:
        thresholds_path = Path(thresholds_path)

    if thresholds_path.exists():
        try:
            with open(thresholds_path, encoding="utf-8") as f:
                data = json.load(f)
            per_class = {k: float(v) for k, v in data.get("thresholds", {}).items()}
            global_th = float(data.get("global_threshold", 0.25))
            return per_class, global_th
        except Exception:
            pass

    return {}, 0.25


def _extract_confidence(det: Any) -> float:
    """Extract float confidence score from detection representation."""
    if hasattr(det, "confidence") and det.confidence is not None:
        return float(det.confidence)
    if hasattr(det, "score") and det.score is not None:
        return float(det.score)
    if isinstance(det, dict):
        if "confidence" in det and det["confidence"] is not None:
            return float(det["confidence"])
        if "score" in det and det["score"] is not None:
            return float(det["score"])
    return 1.0


def _extract_class_name(det: Any) -> str:
    """Extract class name string from detection representation."""
    if hasattr(det, "class_name") and det.class_name:
        return str(det.class_name)
    if isinstance(det, dict) and det.get("class_name"):
        return str(det["class_name"])

    cid = None
    if hasattr(det, "class_id") and det.class_id is not None:
        cid = det.class_id
    elif isinstance(det, dict) and "class_id" in det:
        cid = det["class_id"]

    if cid is not None and cid in CLASS_BY_ID:
        return CLASS_BY_ID[cid].name

    return "Unknown"


def score_image_uncertainty(
    detections_primary: list[Any],
    detections_secondary: list[Any] | None = None,
    thresholds: dict[str, float] | None = None,
    global_threshold: float = 0.25,
    threshold_margin: float = 0.15,
    iou_threshold: float = 0.50,
    weight_threshold: float = 1.0,
    weight_disagreement: float = 1.0,
) -> tuple[float, list[str], float, float, int, int]:
    """
    Score uncertainty of an image's detections.

    Args:
        detections_primary: List of detections from primary model.
        detections_secondary: Optional list of detections from secondary model for disagreement.
        thresholds: Mapping of class name to recommended decision threshold.
        global_threshold: Fallback threshold when class is not in thresholds dict.
        threshold_margin: Distance around threshold considered borderline / uncertain.
        iou_threshold: Overlap threshold for compare bipartite matching.
        weight_threshold: Multiplier for threshold proximity score.
        weight_disagreement: Multiplier for model disagreement score.

    Returns:
        tuple of (
            total_score,
            reasons,
            threshold_score,
            disagreement_score,
            num_near_threshold,
            num_disagreements
        )
    """
    if thresholds is None:
        thresholds = {}

    threshold_score = 0.0
    num_near_th = 0
    near_th_details: list[str] = []

    # 1. Evaluate detections near the threshold
    for det in detections_primary:
        conf = _extract_confidence(det)
        cname = _extract_class_name(det)
        th = thresholds.get(cname, global_threshold)
        delta = abs(conf - th)

        if delta <= threshold_margin:
            # Linear proximity factor: 1.0 at delta=0, decreasing to 0.0 at threshold_margin
            proximity = max(0.0, 1.0 - (delta / threshold_margin))
            threshold_score += proximity
            num_near_th += 1
            near_th_details.append(
                f"{cname} (score {conf:.2f} vs threshold {th:.2f}, Δ={delta:.2f})"
            )

    # 2. Evaluate model disagreement via Day 1 bipartite matching
    disagreement_score = 0.0
    num_disagree = 0
    disagree_details: list[str] = []

    if detections_secondary is not None:
        match_result = match_detections(
            detections_primary,
            detections_secondary,
            iou_threshold=iou_threshold,
            class_aware=True,
        )

        for det_a in match_result.only_a:
            conf_a = _extract_confidence(det_a)
            name_a = _extract_class_name(det_a)
            disagreement_score += conf_a
            num_disagree += 1
            disagree_details.append(f"Model A only: {name_a} ({conf_a:.2f})")

        for det_b in match_result.only_b:
            conf_b = _extract_confidence(det_b)
            name_b = _extract_class_name(det_b)
            disagreement_score += conf_b
            num_disagree += 1
            disagree_details.append(f"Model B only: {name_b} ({conf_b:.2f})")

    total_score = round(
        (threshold_score * weight_threshold)
        + (disagreement_score * weight_disagreement),
        4,
    )

    # 3. Generate explanatory reasons
    reasons: list[str] = []
    if num_near_th > 0:
        details_str = "; ".join(near_th_details[:3])
        if len(near_th_details) > 3:
            details_str += f"; +{len(near_th_details) - 3} more"
        reasons.append(
            f"{num_near_th} detection(s) near decision threshold: {details_str}"
        )

    if num_disagree > 0:
        details_str = "; ".join(disagree_details[:3])
        if len(disagree_details) > 3:
            details_str += f"; +{len(disagree_details) - 3} more"
        reasons.append(f"{num_disagree} model disagreement(s): {details_str}")

    if not reasons:
        reasons.append(
            "High confidence agreement: no detections near threshold or model disagreements"
        )

    return (
        total_score,
        reasons,
        round(threshold_score, 4),
        round(disagreement_score, 4),
        num_near_th,
        num_disagree,
    )


def rank_images_by_uncertainty(
    items: Sequence[ImageUncertaintyInput],
    thresholds: dict[str, float] | None = None,
    global_threshold: float = 0.25,
    threshold_margin: float = 0.15,
    iou_threshold: float = 0.50,
    weight_threshold: float = 1.0,
    weight_disagreement: float = 1.0,
) -> list[RankedImage]:
    """
    Score and rank a batch of candidate images in descending order of uncertainty.

    Args:
        items: Sequence of ImageUncertaintyInput records to rank.
        thresholds: Per-class decision threshold overrides.
        global_threshold: Fallback threshold when class not present.
        threshold_margin: Window around threshold considered uncertain.
        iou_threshold: Matching IoU threshold for disagreement calculation.
        weight_threshold: Multiplier for threshold proximity score.
        weight_disagreement: Multiplier for model disagreement score.

    Returns:
        List of RankedImage sorted from highest uncertainty (rank 1) to lowest.
    """
    if thresholds is None:
        thresholds, loaded_global = load_default_thresholds()
        if global_threshold == 0.25 and loaded_global != 0.25:
            global_threshold = loaded_global

    scored_items: list[
        tuple[float, int, int, str, ImageUncertaintyInput, list[str], float, float]
    ] = []

    for item in items:
        tot_score, reasons, th_score, dis_score, n_th, n_dis = score_image_uncertainty(
            detections_primary=item.detections_primary,
            detections_secondary=item.detections_secondary,
            thresholds=thresholds,
            global_threshold=global_threshold,
            threshold_margin=threshold_margin,
            iou_threshold=iou_threshold,
            weight_threshold=weight_threshold,
            weight_disagreement=weight_disagreement,
        )
        scored_items.append(
            (tot_score, n_dis, n_th, item.image_id, item, reasons, th_score, dis_score)
        )

    # Sort descending by: total score, then disagreements, then near-threshold, then image_id ascending
    scored_items.sort(key=lambda entry: (-entry[0], -entry[1], -entry[2], entry[3]))

    ranked_results: list[RankedImage] = []
    for rank_idx, (
        tot_score,
        n_dis,
        n_th,
        _,
        item,
        reasons,
        th_score,
        dis_score,
    ) in enumerate(scored_items, start=1):
        ranked_results.append(
            RankedImage(
                image_id=item.image_id,
                score=tot_score,
                rank=rank_idx,
                reasons=reasons,
                threshold_uncertainty_score=th_score,
                disagreement_score=dis_score,
                num_near_threshold=n_th,
                num_disagreements=n_dis,
                metadata=dict(item.metadata),
            )
        )

    return ranked_results
