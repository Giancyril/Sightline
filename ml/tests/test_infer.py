"""
ml/tests/test_infer.py
----------------------
Stage 3 Baseline Inference tests.
Verifies COCO SSD model architecture, box decoding, Detector inference,
taxonomy label mapping, CLI execution, and baseline benchmark metrics.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml.eval.baseline import compute_ap
from ml.infer import DetectionResult, Detector
from ml.models.coco_labels import map_coco_id_to_subset
from ml.models.ssd_mobilenet import SSDMobileNetV2, decode_box_offsets


@pytest.mark.stage3
class TestCOCOLabelsMapping:
    def test_shared_coco_classes_map(self) -> None:
        # COCO ID 1 is 'person' -> Subset ID 1 'Person'
        sub_id, sub_name, is_target = map_coco_id_to_subset(1)
        assert is_target is True
        assert sub_id == 1
        assert sub_name == "Person"

        # COCO ID 64 is 'laptop' -> Subset ID 2 'Laptop'
        sub_id, sub_name, is_target = map_coco_id_to_subset(64)
        assert is_target is True
        assert sub_id == 2
        assert sub_name == "Laptop"

    def test_non_subset_coco_class(self) -> None:
        # COCO ID 16 is 'cat' (not in our 20 classes)
        sub_id, sub_name, is_target = map_coco_id_to_subset(16)
        assert is_target is False
        assert sub_id is None
        assert sub_name == "cat"


@pytest.mark.stage3
class TestSSDMobileNetArchitecture:
    def test_forward_shape(self) -> None:
        model = SSDMobileNetV2(num_classes=81)
        model.eval()
        dummy_input = torch.zeros((1, 3, 300, 300), dtype=torch.float32)

        with torch.no_grad():
            offsets, logits = model(dummy_input)

        assert offsets.shape == (1, 2268, 4)
        assert logits.shape == (1, 2268, 81)

    def test_box_offset_decoding(self) -> None:
        anchors = np.array([[0.2, 0.2, 0.4, 0.4]], dtype=np.float32)
        # Offsets [dy=0, dx=0, dh=0, dw=0] -> exactly identical to anchor
        zero_offsets = np.zeros((1, 4), dtype=np.float32)
        decoded = decode_box_offsets(zero_offsets, anchors)
        assert np.allclose(decoded, anchors)

        # Shift dy=+0.1
        shift_offsets = np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
        decoded_shift = decode_box_offsets(shift_offsets, anchors)
        assert decoded_shift[0, 0] > anchors[0, 0]  # ymin shifted down


@pytest.mark.stage3
class TestDetectorInference:
    def test_detector_predict_and_annotate(self, tmp_path: Path) -> None:
        detector = Detector(confidence_threshold=0.10)
        test_img = Image.new("RGB", (640, 480), (100, 150, 200))

        result = detector.predict(test_img)
        assert isinstance(result, DetectionResult)
        assert result.image_size == {"width": 640, "height": 480}
        assert result.inference_ms > 0.0
        assert isinstance(result.detections, list)

        # Test annotate
        annotated = detector.annotate(test_img, result.detections)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == (640, 480)

    def test_compute_ap_metric(self) -> None:
        gt = [[0.1, 0.1, 0.5, 0.5]]
        preds = [[0.12, 0.12, 0.48, 0.48]]
        scores = [0.9]
        ap = compute_ap(preds, scores, gt, iou_threshold=0.5)
        assert ap == pytest.approx(1.0)


@pytest.mark.stage3
class TestBaselineMetricsAndCLI:
    def test_baseline_metrics_recorded(self) -> None:
        metrics_path = Path("docs/baseline_metrics.json")
        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text(encoding="utf-8"))

        assert "mAP_coco_shared" in data
        assert "mAP_novel_classes" in data
        assert data["mAP_novel_classes"] == 0.0  # Novel classes must be 0 for baseline
        assert (
            data["mAP_coco_shared"] > 0.20
        )  # Baseline retains pre-trained shared ability
        assert data["target_stage5_goals"]["novel_classes_mAP_target"] >= 0.10

    def test_cli_execution(self, tmp_path: Path) -> None:
        # Create a test sample image
        test_img_path = tmp_path / "cli_test.jpg"
        img = Image.new("RGB", (300, 300), (200, 200, 200))
        img.save(test_img_path)

        out_json = tmp_path / "out.json"
        out_img = tmp_path / "out.jpg"

        cmd = [
            sys.executable,
            "-m",
            "ml.infer",
            str(test_img_path),
            "--threshold",
            "0.2",
            "--output-json",
            str(out_json),
            "--output-image",
            str(out_img),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert out_json.exists()
        assert out_img.exists()
        # Verify JSON content
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert "inference_ms" in data
        assert "detections" in data
