"""
ml/tests/test_onnx_export.py
----------------------------
Unit tests for ONNX export and ONNX Runtime inference.
Validates ONNX graph validation, session execution, numerical parity
with PyTorch float32 baseline, and detection pipeline decoding.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from PIL import Image

from ml.export.onnx_export import (
    ONNXDetector,
    export_onnx_model,
    verify_onnx_parity,
)
from ml.models.ssd_mobilenet import SSDMobileNetV2


@pytest.fixture
def small_model() -> SSDMobileNetV2:
    """Fixture model with 21 classes."""
    torch.manual_seed(42)
    model = SSDMobileNetV2(num_classes=21)
    model.eval()
    return model


def test_export_onnx_model_creates_valid_file(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "model_weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_onnx = Path(tmp_dir) / "model.onnx"
        meta = export_onnx_model(
            checkpoint_path=tmp_pth,
            output_path=out_onnx,
            num_classes=21,
            opset_version=18,
        )

        assert out_onnx.exists()
        assert out_onnx.stat().st_size > 0
        assert meta["format"] == "onnx"
        assert meta["opset_version"] == 18

        # Verify with ONNX checker
        loaded_model = onnx.load(str(out_onnx))
        onnx.checker.check_model(loaded_model)
        assert len(loaded_model.graph.input) == 1
        assert len(loaded_model.graph.output) == 2


def test_onnx_runtime_inference_shapes(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "model_weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_onnx = Path(tmp_dir) / "model.onnx"
        export_onnx_model(
            checkpoint_path=tmp_pth,
            output_path=out_onnx,
            num_classes=21,
        )

        detector = ONNXDetector(out_onnx)
        dummy_x = np.random.randn(1, 3, 300, 300).astype(np.float32)
        locs, scores = detector.forward(dummy_x)

        assert locs.shape == (1, 2268, 4)
        assert scores.shape == (1, 2268, 21)
        assert locs.dtype == np.float32
        assert scores.dtype == np.float32


def test_onnx_parity_with_pytorch(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "model_weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_onnx = Path(tmp_dir) / "model.onnx"
        export_onnx_model(
            checkpoint_path=tmp_pth,
            output_path=out_onnx,
            num_classes=21,
        )

        detector = ONNXDetector(out_onnx)
        test_inputs = [
            np.random.randn(1, 3, 300, 300).astype(np.float32),
            np.zeros((1, 3, 300, 300), dtype=np.float32),
            np.ones((1, 3, 300, 300), dtype=np.float32) * 0.5,
        ]

        verification = verify_onnx_parity(
            original_model=small_model,
            onnx_detector=detector,
            test_inputs=test_inputs,
            atol=1e-3,
        )

        assert verification["passed"] is True
        assert verification["max_loc_difference"] < 1e-3
        assert verification["max_score_difference"] < 1e-3


def test_onnx_detector_predict_pipeline(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "model_weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_onnx = Path(tmp_dir) / "model.onnx"
        export_onnx_model(
            checkpoint_path=tmp_pth,
            output_path=out_onnx,
            num_classes=21,
        )

        detector = ONNXDetector(out_onnx, confidence_threshold=0.1)
        img = Image.new("RGB", (640, 480), color=(120, 160, 200))
        dets = detector.predict(img)

        assert isinstance(dets, list)
        for box, cid, score in dets:
            assert len(box) == 4
            assert 0 <= cid < 21
            assert score >= 0.1
