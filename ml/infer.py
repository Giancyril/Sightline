"""
ml/infer.py
-----------
Detector class and CLI interface for SSD object detection.
Handles image preprocessing, model prediction, anchor decoding,
confidence thresholding, class-wise NMS, class mapping, and annotated rendering.

Usage:
    python -m ml.infer <image_path> [--threshold 0.3] [--output-image <path>] [--output-json <path>]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ml.data.transforms import generate_ssd_anchors, normalize_image
from ml.models.coco_labels import map_coco_id_to_subset
from ml.models.ssd_mobilenet import (
    SSDMobileNetV2,
    postprocess_ssd_predictions,
)
from ml.utils.drawing import draw_bounding_boxes


@dataclass
class Detection:
    label: str
    label_id: int
    score: float
    box: list[float]  # Normalized [ymin, xmin, ymax, xmax] in [0, 1]
    box_px: list[int]  # Pixel [ymin, xmin, ymax, xmax]
    is_coco_class: bool


@dataclass
class DetectionResult:
    image_path: str
    image_size: dict[str, int]
    inference_ms: float
    detections: list[Detection]
    model_name: str
    threshold: float


class Detector:
    """
    Object detector wrapping SSD MobileNet V2 inference.
    Executes preprocessing, inference, NMS, and class taxonomy mapping.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_size: tuple[int, int] = (300, 300),
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.device = torch.device(device) if "torch" in globals() else "cpu"
        self.model_name = "SSD-MobileNetV2-COCO-Baseline"

        # Generate anchors once
        self.anchors = generate_ssd_anchors()

        # Initialize SSD MobileNet V2 architecture
        self.model = SSDMobileNetV2(num_classes=81)
        if model_path is not None and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    def preprocess(self, pil_img: Image.Image) -> tuple[Any, tuple[int, int]]:
        """Resize and normalize PIL image to model input tensor."""
        orig_w, orig_h = pil_img.size
        resized = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32)
        norm_img = normalize_image(img_np)
        # (H, W, C) -> (1, C, H, W)
        tensor = (
            torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        return tensor, (orig_w, orig_h)

    def predict(
        self,
        image: Image.Image | np.ndarray | Path | str,
        threshold: float | None = None,
    ) -> DetectionResult:
        """Run full object detection inference on an image."""
        conf_thresh = threshold if threshold is not None else self.confidence_threshold

        if isinstance(image, (str, Path)):
            img_path = str(image)
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img_path = "in_memory_array.jpg"
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            img_path = "in_memory_image.jpg"
            pil_img = image.convert("RGB")

        orig_w, orig_h = pil_img.size

        # Preprocess
        tensor, _ = self.preprocess(pil_img)

        # Inference with timing
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_offsets, pred_logits = self.model(tensor)
            pred_probs = torch.softmax(pred_logits, dim=-1)[0].cpu().numpy()
            pred_offsets = pred_offsets[0].cpu().numpy()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Postprocess via NMS
        raw_detections = postprocess_ssd_predictions(
            pred_offsets,
            pred_probs,
            self.anchors,
            confidence_threshold=conf_thresh,
            iou_threshold=self.iou_threshold,
        )

        detections: list[Detection] = []
        for box, raw_cls_id, score in raw_detections:
            subset_id, subset_name, is_target = map_coco_id_to_subset(raw_cls_id)
            if not is_target or subset_id is None or subset_name is None:
                continue

            ymin, xmin, ymax, xmax = box
            box_px = [
                int(round(ymin * orig_h)),
                int(round(xmin * orig_w)),
                int(round(ymax * orig_h)),
                int(round(xmax * orig_w)),
            ]

            det = Detection(
                label=subset_name,
                label_id=subset_id,
                score=round(score, 4),
                box=[round(c, 5) for c in box],
                box_px=box_px,
                is_coco_class=True,
            )
            detections.append(det)

        return DetectionResult(
            image_path=img_path,
            image_size={"width": orig_w, "height": orig_h},
            inference_ms=round(inference_ms, 2),
            detections=detections,
            model_name=self.model_name,
            threshold=conf_thresh,
        )

    def annotate(
        self,
        image: Image.Image | np.ndarray,
        detections: list[Detection],
    ) -> Image.Image:
        """Render colored bounding boxes and chips on image."""
        boxes = [d.box for d in detections]
        labels = [d.label_id for d in detections]
        scores = [d.score for d in detections]
        return draw_bounding_boxes(image, boxes, labels, scores=scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="SSD MobileNet V2 Inference CLI")
    parser.add_argument("image_path", type=Path, help="Path to input image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence score threshold",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Path to save annotated output image",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save output JSON",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Input image not found: {args.image_path}")

    detector = Detector(confidence_threshold=args.threshold)
    result = detector.predict(args.image_path, threshold=args.threshold)

    # Print JSON output to stdout
    result_dict = asdict(result)
    json_output = json.dumps(result_dict, indent=2)
    print(json_output)

    # Save JSON if requested
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_output, encoding="utf-8")
        print(f"Saved detection JSON to: {args.output_json}")

    # Generate and save annotated image
    out_img_path = args.output_image
    if out_img_path is None:
        out_img_path = args.image_path.parent / f"{args.image_path.stem}_detected.jpg"

    input_img = Image.open(args.image_path).convert("RGB")
    annotated = detector.annotate(input_img, result.detections)
    annotated.save(out_img_path, "JPEG")
    print(f"Saved annotated image to: {out_img_path}")


if __name__ == "__main__":
    main()
