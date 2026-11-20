"""
ml/tests/test_active_learning.py
--------------------------------
Unit tests for Day 4 Commit 8:
  - Active learning uncertainty ranking for images
  - Score formulation: detections near threshold + model disagreement (unmatched boxes)
  - Explanatory reasons for each ranked image
  - Deterministic ordering and edge cases (empty inputs, custom thresholds, tie breaking)
"""

from __future__ import annotations

from ml.eval.active_learning import (
    ImageUncertaintyInput,
    rank_images_by_uncertainty,
    score_image_uncertainty,
)


def _make_box(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5):
    return [ymin, xmin, ymax, xmax]


def test_ordering_near_threshold_vs_confident():
    """An image with detections near the threshold must rank higher than an image with confident detections."""
    # Image A: Borderline detection (0.26 vs threshold 0.25 -> delta 0.01)
    img_borderline = ImageUncertaintyInput(
        image_id="img_borderline",
        detections_primary=[
            {
                "class_name": "Person",
                "class_id": 1,
                "confidence": 0.26,
                "box": _make_box(),
            }
        ],
    )

    # Image B: High confidence detection (0.95 vs threshold 0.25 -> delta 0.70 > margin 0.15)
    img_confident = ImageUncertaintyInput(
        image_id="img_confident",
        detections_primary=[
            {
                "class_name": "Person",
                "class_id": 1,
                "confidence": 0.95,
                "box": _make_box(),
            }
        ],
    )

    ranked = rank_images_by_uncertainty(
        [img_confident, img_borderline],
        global_threshold=0.25,
        threshold_margin=0.15,
    )

    assert len(ranked) == 2
    assert ranked[0].image_id == "img_borderline"
    assert ranked[0].rank == 1
    assert ranked[0].score > ranked[1].score
    assert ranked[1].image_id == "img_confident"
    assert ranked[1].rank == 2
    assert ranked[1].score == 0.0


def test_ordering_disagreement_vs_agreement():
    """Model disagreement produces high uncertainty, outranking full model agreement."""
    # Image Agreement: Model A and Model B detect the same object with high IoU
    img_agreement = ImageUncertaintyInput(
        image_id="img_agree",
        detections_primary=[
            {
                "class_name": "Laptop",
                "confidence": 0.90,
                "box": [0.2, 0.2, 0.6, 0.6],
            }
        ],
        detections_secondary=[
            {
                "class_name": "Laptop",
                "confidence": 0.88,
                "box": [0.21, 0.21, 0.61, 0.61],
            }
        ],
    )

    # Image Disagreement: Model A detects Laptop, but Model B detects nothing
    img_disagreement = ImageUncertaintyInput(
        image_id="img_disagree",
        detections_primary=[
            {
                "class_name": "Laptop",
                "confidence": 0.90,
                "box": [0.2, 0.2, 0.6, 0.6],
            }
        ],
        detections_secondary=[],
    )

    ranked = rank_images_by_uncertainty([img_agreement, img_disagreement])

    assert ranked[0].image_id == "img_disagree"
    assert ranked[0].rank == 1
    assert ranked[0].num_disagreements == 1
    assert ranked[0].disagreement_score == 0.90
    assert ranked[1].image_id == "img_agree"
    assert ranked[1].rank == 2
    assert ranked[1].num_disagreements == 0


def test_combined_ranking_multiple_images():
    """Verify strict multi-image ordering across compound uncertainty scenarios."""
    # 1. Clean agreement + confident
    clean = ImageUncertaintyInput(
        image_id="img_clean",
        detections_primary=[{"class_name": "Car", "confidence": 0.92, "box": [0.1, 0.1, 0.4, 0.4]}],
        detections_secondary=[{"class_name": "Car", "confidence": 0.90, "box": [0.1, 0.1, 0.4, 0.4]}],
    )

    # 2. Near-threshold detection only
    near_th = ImageUncertaintyInput(
        image_id="img_near_th",
        detections_primary=[{"class_name": "Chair", "confidence": 0.28, "box": [0.1, 0.1, 0.4, 0.4]}],
        detections_secondary=[{"class_name": "Chair", "confidence": 0.28, "box": [0.1, 0.1, 0.4, 0.4]}],
    )

    # 3. Model disagreement only (Model B predicts another object)
    disagree = ImageUncertaintyInput(
        image_id="img_disagree",
        detections_primary=[{"class_name": "Dog", "confidence": 0.85, "box": [0.1, 0.1, 0.4, 0.4]}],
        detections_secondary=[
            {"class_name": "Dog", "confidence": 0.85, "box": [0.1, 0.1, 0.4, 0.4]},
            {"class_name": "Person", "confidence": 0.80, "box": [0.5, 0.5, 0.9, 0.9]},  # only in B
        ],
    )

    # 4. Compound: Both model disagreement AND near-threshold boxes
    compound = ImageUncertaintyInput(
        image_id="img_compound",
        detections_primary=[
            {"class_name": "Chair", "confidence": 0.25, "box": [0.1, 0.1, 0.4, 0.4]},  # delta=0.0 -> max th score 1.0
            {"class_name": "Person", "confidence": 0.85, "box": [0.5, 0.5, 0.8, 0.8]},
        ],
        detections_secondary=[
            {"class_name": "Chair", "confidence": 0.25, "box": [0.1, 0.1, 0.4, 0.4]},
            {"class_name": "Book", "confidence": 0.75, "box": [0.7, 0.7, 0.95, 0.95]},  # disagreement in A and B
        ],
    )

    ranked = rank_images_by_uncertainty([clean, near_th, disagree, compound], global_threshold=0.25)

    assert [r.image_id for r in ranked] == [
        "img_compound",
        "img_disagree",
        "img_near_th",
        "img_clean",
    ]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]


def test_reasons_returned_for_each_rank():
    """Verify detailed reasons are populated describing both threshold and disagreement causes."""
    item = ImageUncertaintyInput(
        image_id="img_reasons",
        detections_primary=[
            {"class_name": "Person", "confidence": 0.27, "box": [0.1, 0.1, 0.5, 0.5]},
        ],
        detections_secondary=[
            {"class_name": "Laptop", "confidence": 0.78, "box": [0.6, 0.6, 0.9, 0.9]},
        ],
    )

    score, reasons, th_score, dis_score, n_th, n_dis = score_image_uncertainty(
        detections_primary=item.detections_primary,
        detections_secondary=item.detections_secondary,
        global_threshold=0.25,
        threshold_margin=0.15,
    )

    assert score > 0.0
    assert len(reasons) >= 2
    # Threshold reason
    assert any("near decision threshold" in r for r in reasons)
    assert any("Person" in r for r in reasons)
    # Disagreement reason
    assert any("model disagreement" in r for r in reasons)
    assert n_th == 1
    assert n_dis == 2  # Person only in A, Laptop only in B


def test_per_class_thresholds_applied():
    """Verify that Day 3 per-class thresholds override global threshold for calculating proximity."""
    # Suppose Guitar has recommended threshold 0.35, while global threshold is 0.25
    guitar_det = [{"class_name": "Guitar", "confidence": 0.36, "box": [0.1, 0.1, 0.5, 0.5]}]

    # With per-class threshold (0.35), confidence 0.36 has delta 0.01 (high threshold uncertainty)
    score_custom, _, th_custom, _, n_th_custom, _ = score_image_uncertainty(
        detections_primary=guitar_det,
        thresholds={"Guitar": 0.35},
        global_threshold=0.25,
        threshold_margin=0.15,
    )
    assert n_th_custom == 1
    assert th_custom > 0.90  # Proximity is (1 - 0.01/0.15) ~= 0.933

    # With global threshold (0.25), confidence 0.36 has delta 0.11 (lower proximity)
    score_global, _, th_global, _, n_th_global, _ = score_image_uncertainty(
        detections_primary=guitar_det,
        thresholds={},
        global_threshold=0.25,
        threshold_margin=0.15,
    )
    assert n_th_global == 1
    assert th_global < 0.35
    assert th_custom > th_global


def test_empty_detections_clean_reason():
    """Empty detections produce score 0.0 and clear high confidence agreement reason."""
    item = ImageUncertaintyInput(image_id="img_empty", detections_primary=[], detections_secondary=[])
    score, reasons, _, _, n_th, n_dis = score_image_uncertainty(
        detections_primary=item.detections_primary,
        detections_secondary=item.detections_secondary,
    )
    assert score == 0.0
    assert n_th == 0
    assert n_dis == 0
    assert len(reasons) == 1
    assert "High confidence agreement" in reasons[0]


def test_deterministic_tie_breaker():
    """Equal scores break ties predictably using disagreements, near-th count, and image_id."""
    img_b = ImageUncertaintyInput(image_id="img_b", detections_primary=[])
    img_a = ImageUncertaintyInput(image_id="img_a", detections_primary=[])

    ranked = rank_images_by_uncertainty([img_b, img_a])
    assert ranked[0].image_id == "img_a"
    assert ranked[1].image_id == "img_b"
