"""
ml/eval/dump_predictions.py
---------------------------
Extract and dump model validation predictions and ground truth to a versioned
compressed file (.npz) for threshold selection and score calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml.data.schema import read_manifest_csv
from ml.models.registry import get_model_entry

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(file_path)
    if not p.exists():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump_validation_predictions(
    model_id: str = "fine-tuned",
    manifest_path: str | Path = "ml/data/subset/manifest_val.csv",
    output_path: str | Path | None = None,
    conf_threshold: float = 0.01,
    nms_iou_threshold: float = 0.45,
    detector: Any = None,
    registry_path: str | Path | None = None,
) -> Path:
    """
    Run model detection on validation images from manifest and save all
    candidate predictions and ground truth annotations into a compressed .npz.

    Args:
        model_id: Model ID registered in models/registry.json.
        manifest_path: Path to validation manifest CSV.
        output_path: Destination .npz path. If None, defaults to ml/eval/dumps/<model_id>_val_predictions.npz.
        conf_threshold: Low confidence cutoff to capture raw score distribution.
        nms_iou_threshold: NMS overlap threshold.
        detector: Optional pre-instantiated detector instance (or mock). If None, loaded via get_detector.
        registry_path: Optional path to registry.json.

    Returns:
        Path to the saved .npz file.
    """
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_p}")

    model_entry = get_model_entry(model_id, registry_path=registry_path)
    model_version = model_entry.version if model_entry else "unknown"

    dataset_hash = compute_file_sha256(manifest_p)
    timestamp = datetime.now(UTC).isoformat()

    if output_path is None:
        out_p = Path("ml/eval/dumps") / f"{model_id}_val_predictions.npz"
    else:
        out_p = Path(output_path)

    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read manifest records
    records = read_manifest_csv(manifest_p)

    # 2. Setup detector if not passed
    if detector is None:
        from api.services.detector import get_detector  # noqa: PLC0415

        detector = get_detector(model_id)

    # 3. Accumulate detections and ground truths
    det_image_ids: list[str] = []
    det_class_ids: list[int] = []
    det_class_names: list[str] = []
    det_scores: list[float] = []
    det_boxes: list[list[float]] = []

    gt_image_ids: list[str] = []
    gt_class_ids: list[int] = []
    gt_class_names: list[str] = []
    gt_boxes: list[list[float]] = []
    gt_is_group_of: list[int] = []
    gt_is_occluded: list[int] = []
    gt_is_truncated: list[int] = []

    for img_rec in records:
        img_id = img_rec.image_id

        # Ground truths for this image
        for ann in img_rec.annotations:
            gt_image_ids.append(img_id)
            gt_class_ids.append(ann.label_id)
            gt_class_names.append(ann.label)
            gt_boxes.append([ann.ymin, ann.xmin, ann.ymax, ann.xmax])
            gt_is_group_of.append(int(ann.is_group_of))
            gt_is_occluded.append(int(ann.is_occluded))
            gt_is_truncated.append(int(ann.is_truncated))

        # Run inference if image exists
        img_path = Path(img_rec.file_path)
        if not img_path.exists():
            continue

        try:
            img_bytes = img_path.read_bytes()
            res = detector.detect(
                image_bytes=img_bytes,
                confidence_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
            for d in res.get("detections", []):
                det_image_ids.append(img_id)
                det_class_ids.append(d.get("class_id", 0))
                det_class_names.append(d.get("class_name", ""))
                det_scores.append(float(d.get("confidence", 0.0)))
                box = d.get("box_normalized", {})
                det_boxes.append(
                    [
                        float(box.get("ymin", 0.0)),
                        float(box.get("xmin", 0.0)),
                        float(box.get("ymax", 0.0)),
                        float(box.get("xmax", 0.0)),
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inference failed on %s: %s", img_path, exc)

    metadata = {
        "model_id": model_id,
        "model_version": model_version,
        "dataset_hash": dataset_hash,
        "date": timestamp,
        "manifest_path": str(manifest_p),
        "conf_threshold": conf_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "num_detections": len(det_scores),
        "num_ground_truth": len(gt_class_ids),
        "num_images": len(records),
    }

    # Format arrays
    det_boxes_arr = (
        np.array(det_boxes, dtype=np.float32)
        if det_boxes
        else np.zeros((0, 4), dtype=np.float32)
    )
    gt_boxes_arr = (
        np.array(gt_boxes, dtype=np.float32)
        if gt_boxes
        else np.zeros((0, 4), dtype=np.float32)
    )

    np.savez_compressed(
        out_p,
        metadata_json=np.array(json.dumps(metadata)),
        det_image_ids=np.array(det_image_ids, dtype=object),
        det_class_ids=np.array(det_class_ids, dtype=np.int32),
        det_class_names=np.array(det_class_names, dtype=object),
        det_scores=np.array(det_scores, dtype=np.float32),
        det_boxes=det_boxes_arr,
        gt_image_ids=np.array(gt_image_ids, dtype=object),
        gt_class_ids=np.array(gt_class_ids, dtype=np.int32),
        gt_class_names=np.array(gt_class_names, dtype=object),
        gt_boxes=gt_boxes_arr,
        gt_is_group_of=np.array(gt_is_group_of, dtype=np.int32),
        gt_is_occluded=np.array(gt_is_occluded, dtype=np.int32),
        gt_is_truncated=np.array(gt_is_truncated, dtype=np.int32),
    )

    logger.info(
        "Saved %d detections and %d GTs to %s",
        len(det_scores),
        len(gt_class_ids),
        out_p,
    )
    return out_p


def load_predictions_dump(dump_path: str | Path) -> dict[str, Any]:
    """
    Load a versioned predictions dump .npz file.

    Returns:
        dict containing metadata, detections dict, and ground_truth dict.
    """
    p = Path(dump_path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions dump not found: {p}")

    with np.load(p, allow_pickle=True) as data:
        metadata_str = str(data["metadata_json"])
        metadata = json.loads(metadata_str)

        detections = {
            "image_ids": np.array(data["det_image_ids"], dtype=str),
            "class_ids": np.array(data["det_class_ids"], dtype=np.int32),
            "class_names": np.array(data["det_class_names"], dtype=str),
            "scores": np.array(data["det_scores"], dtype=np.float32),
            "boxes": np.array(data["det_boxes"], dtype=np.float32),
        }

        ground_truth = {
            "image_ids": np.array(data["gt_image_ids"], dtype=str),
            "class_ids": np.array(data["gt_class_ids"], dtype=np.int32),
            "class_names": np.array(data["gt_class_names"], dtype=str),
            "boxes": np.array(data["gt_boxes"], dtype=np.float32),
            "is_group_of": np.array(data["gt_is_group_of"], dtype=np.int32),
            "is_occluded": np.array(data["gt_is_occluded"], dtype=np.int32),
            "is_truncated": np.array(data["gt_is_truncated"], dtype=np.int32),
        }

    return {
        "metadata": metadata,
        "detections": detections,
        "ground_truth": ground_truth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump validation predictions to versioned file."
    )
    parser.add_argument(
        "--model-id", default="fine-tuned", help="Model ID from registry."
    )
    parser.add_argument(
        "--val-manifest",
        default="ml/data/subset/manifest_val.csv",
        help="Val manifest.",
    )
    parser.add_argument("--output", default=None, help="Output .npz file.")
    parser.add_argument(
        "--conf-threshold", type=float, default=0.01, help="Score threshold."
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.45, help="NMS threshold."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dump_validation_predictions(
        model_id=args.model_id,
        manifest_path=args.val_manifest,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        nms_iou_threshold=args.nms_threshold,
    )


if __name__ == "__main__":
    main()
