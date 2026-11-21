"""
ml/tests/test_int8_quantization.py
----------------------------------
Unit tests for full-integer INT8 quantization export and representative dataset.
Verifies calibration sampling from configuration, INT8/INT32/UINT8 tensor shapes
and dtypes, TFLite container serialization, and quantized inference execution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml.export.tflite_export import TFLITE_MAGIC
from ml.export.tflite_full_integer import (
    TFLiteFullIntegerDetector,
    export_tflite_full_integer,
    get_representative_dataset_generator,
    quantize_full_integer,
)
from ml.models.ssd_mobilenet import SSDMobileNetV2


@pytest.fixture
def small_model() -> SSDMobileNetV2:
    """Fixture model with 21 classes."""
    torch.manual_seed(42)
    model = SSDMobileNetV2(num_classes=21)
    model.eval()
    return model


def test_representative_dataset_generator_from_config() -> None:
    config_path = "ml/train/config.yaml"
    num_samples = 5
    gen = get_representative_dataset_generator(config_path, num_samples=num_samples)
    samples = list(gen)

    assert len(samples) == num_samples
    for s in samples:
        assert isinstance(s, np.ndarray)
        assert s.shape == (1, 3, 300, 300)
        assert s.dtype == np.float32
        assert s.min() >= -1.05
        assert s.max() <= 1.05


def test_representative_dataset_synthetic_fallback() -> None:
    # Fallback when config is missing
    gen = get_representative_dataset_generator(config_path=None, num_samples=3)
    samples = list(gen)

    assert len(samples) == 3
    for s in samples:
        assert s.shape == (1, 3, 300, 300)
        assert s.dtype == np.float32
        assert s.min() >= -1.0
        assert s.max() <= 1.0


def test_full_integer_quantization_shapes_and_dtypes(
    small_model: SSDMobileNetV2,
) -> None:
    gen = get_representative_dataset_generator(num_samples=3)
    samples = list(gen)

    quant_weights, meta = quantize_full_integer(small_model, samples, num_classes=21)

    # 1. Verify schema tensor specs
    specs = meta["tensor_specs"]
    assert specs["input"]["shape"] == [1, 3, 300, 300]
    assert specs["input"]["dtype"] == "int8"
    assert specs["input"]["scale"] > 0
    assert specs["input"]["zero_point"] == 0

    assert specs["output_locations"]["shape"] == [1, 2268, 4]
    assert specs["output_locations"]["dtype"] == "int8"
    assert specs["output_locations"]["scale"] > 0

    assert specs["output_scores"]["shape"] == [1, 2268, 21]
    assert specs["output_scores"]["dtype"] == "uint8"
    assert specs["output_scores"]["scale"] > 0

    # 2. Verify quantized weight and bias array dtypes
    for name, item in quant_weights.items():
        if "weight" in name and "stem.0" in name:
            assert item["dtype"] == "int8"
            assert item["data"].dtype == np.int8
            assert len(item["shape"]) == 4
            assert item["scale"] > 0
        elif "cls_heads.0.bias" in name:
            assert item["dtype"] == "int32"
            assert item["data"].dtype == np.int32
            assert item["scale"] > 0


def test_export_tflite_full_integer_file(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_tflite = Path(tmp_dir) / "full_int.tflite"
        meta = export_tflite_full_integer(
            checkpoint_path=tmp_pth,
            config_path="ml/train/config.yaml",
            output_path=out_tflite,
            num_samples=3,
            num_classes=21,
        )

        assert out_tflite.exists()
        raw = out_tflite.read_bytes()
        assert raw[4:8] == TFLITE_MAGIC
        assert meta["format"] == "tflite_full_integer"
        assert meta["file_size_bytes"] == len(raw)
        assert meta["compression"]["compression_ratio"] >= 2.5


def test_full_integer_detector_inference(small_model: SSDMobileNetV2) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pth = Path(tmp_dir) / "weights.pth"
        torch.save(small_model.state_dict(), tmp_pth)

        out_tflite = Path(tmp_dir) / "full_int.tflite"
        export_tflite_full_integer(
            checkpoint_path=tmp_pth,
            config_path="ml/train/config.yaml",
            output_path=out_tflite,
            num_samples=2,
            num_classes=21,
        )

        detector = TFLiteFullIntegerDetector(out_tflite, confidence_threshold=0.1)

        # 1. Test forward_quantized with int8 tensor
        int8_input = np.zeros((1, 3, 300, 300), dtype=np.int8)
        q_locs, q_scores = detector.forward_quantized(int8_input)

        assert q_locs.shape == (1, 2268, 4)
        assert q_locs.dtype == np.int8

        assert q_scores.shape == (1, 2268, 21)
        assert q_scores.dtype == np.uint8

        # 2. Test predict with PIL image
        dummy_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        dets = detector.predict(dummy_img)
        assert isinstance(dets, list)
        for box, cid, score in dets:
            assert len(box) == 4
            assert 0 <= cid < 21
            assert score >= 0.1
