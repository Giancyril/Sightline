"""
ml/tests/test_tflite_export.py
------------------------------
Unit tests for TFLite dynamic-range quantization export and verification.
Validates weight quantization, TFLite container serialization,
and output parity against original PyTorch float32 model within tolerance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml.export.tflite_export import (
    TFLITE_MAGIC,
    TFLiteDynamicRangeDetector,
    deserialize_tflite_container,
    export_tflite_dynamic_range,
    quantize_weights_dynamic_range,
    serialize_tflite_container,
    verify_dynamic_range_parity,
)
from ml.models.ssd_mobilenet import SSDMobileNetV2


@pytest.fixture
def small_model() -> SSDMobileNetV2:
    """Fixture model with 21 classes."""
    torch.manual_seed(42)
    model = SSDMobileNetV2(num_classes=21)
    model.eval()
    return model


def test_quantize_weights_dynamic_range(small_model: SSDMobileNetV2) -> None:
    state = small_model.state_dict()
    quantized, meta = quantize_weights_dynamic_range(state)

    assert meta["num_tensors"] == len(state)
    assert meta["quantized_weight_tensors"] > 0
    assert meta["compression_ratio"] >= 2.5
    assert meta["size_reduction_pct"] > 60.0

    # Verify conv weight is int8
    for name, item in quantized.items():
        if "weight" in name and "stem.0" in name:
            assert item["dtype"] == "int8"
            assert "scale" in item
            assert item["scale"] > 0
            assert isinstance(item["data"], np.ndarray)
            assert item["data"].dtype == np.int8


def test_serialize_deserialize_tflite_container() -> None:
    metadata = {
        "model_name": "test_ssd",
        "num_classes": 21,
        "format": "tflite_dynamic_range",
    }
    dummy_weights = {
        "layer1.weight": {
            "dtype": "int8",
            "scale": 0.005,
            "shape": [16, 3, 3, 3],
            "data": np.random.randint(-128, 127, size=(16, 3, 3, 3), dtype=np.int8),
        },
        "layer1.bias": {
            "dtype": "float32",
            "scale": 1.0,
            "shape": [16],
            "data": np.random.randn(16).astype(np.float32),
        },
    }

    binary = serialize_tflite_container(dummy_weights, metadata)
    assert binary[4:8] == TFLITE_MAGIC

    recovered_meta, recovered_weights = deserialize_tflite_container(binary)
    assert recovered_meta["model_name"] == "test_ssd"
    assert "layer1.weight" in recovered_weights
    assert recovered_weights["layer1.weight"]["dtype"] == "int8"
    assert recovered_weights["layer1.weight"]["scale"] == pytest.approx(0.005, abs=1e-5)
    np.testing.assert_array_equal(
        recovered_weights["layer1.weight"]["data"],
        dummy_weights["layer1.weight"]["data"],
    )


def test_export_tflite_dynamic_range(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "test_weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_tflite = Path(tmp_dir) / "model_dr.tflite"
        result = export_tflite_dynamic_range(
            checkpoint_path=tmp_pth,
            output_path=out_tflite,
            num_classes=21,
        )

        assert out_tflite.exists()
        assert out_tflite.stat().st_size > 0
        raw_header = out_tflite.read_bytes()[:8]
        assert raw_header[4:8] == TFLITE_MAGIC

        assert result["format"] == "tflite_dynamic_range"
        assert result["file_size_bytes"] == out_tflite.stat().st_size
        assert result["metrics"]["compression_ratio"] >= 2.5


def test_verify_outputs_match_original_within_tolerance(
    small_model: SSDMobileNetV2,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "test_weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_tflite = Path(tmp_dir) / "model_dr.tflite"
        export_tflite_dynamic_range(
            checkpoint_path=tmp_pth,
            output_path=out_tflite,
            num_classes=21,
        )

        detector = TFLiteDynamicRangeDetector(out_tflite)

        # Test on multiple inputs: zero tensor, random tensor, fixed value
        torch.manual_seed(99)
        test_inputs = [
            torch.randn(1, 3, 300, 300),
            torch.zeros(1, 3, 300, 300),
            torch.ones(1, 3, 300, 300) * 0.25,
        ]

        verification = verify_dynamic_range_parity(
            original_model=small_model,
            quantized_detector=detector,
            test_inputs=test_inputs,
            atol=0.05,
            rtol=0.05,
        )

        assert verification["passed"] is True
        assert verification["max_loc_difference"] < 0.05
        assert verification["min_cosine_similarity"] > 0.99


def test_tflite_detector_predict_fixture_image() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = SSDMobileNetV2(num_classes=21)
        tmp_pth = Path(tmp_dir) / "test_weights.pth"
        torch.save(model.state_dict(), tmp_pth)

        out_tflite = Path(tmp_dir) / "model_dr.tflite"
        export_tflite_dynamic_range(
            checkpoint_path=tmp_pth,
            output_path=out_tflite,
            num_classes=21,
        )

        detector = TFLiteDynamicRangeDetector(out_tflite, confidence_threshold=0.1)

        # Synthetic fixture image
        dummy_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        dets = detector.predict(dummy_img)
        assert isinstance(dets, list)
        for box, cid, score in dets:
            assert len(box) == 4
            assert 0 <= cid < 21
            assert score >= 0.1
