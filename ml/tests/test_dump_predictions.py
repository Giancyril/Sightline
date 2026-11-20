"""
ml/tests/test_dump_predictions.py
---------------------------------
Unit tests for validation predictions dumping and loading (Commit 1).
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.eval.dump_predictions import (
    compute_file_sha256,
    dump_validation_predictions,
    load_predictions_dump,
)

FIXTURE_PATH = Path("ml/tests/fixtures/val_dump_fixture.npz")


def test_load_predictions_dump_fixture():
    """Verify loading pre-generated fixture dump."""
    assert FIXTURE_PATH.exists(), f"Fixture missing at {FIXTURE_PATH}"

    data = load_predictions_dump(FIXTURE_PATH)
    assert "metadata" in data
    assert "detections" in data
    assert "ground_truth" in data

    meta = data["metadata"]
    assert meta["model_id"] == "fine-tuned"
    assert meta["model_version"] == "v1.0.0"
    assert len(meta["dataset_hash"]) == 64
    assert meta["conf_threshold"] == 0.01
    assert meta["num_detections"] == 10
    assert meta["num_ground_truth"] == 6

    dets = data["detections"]
    assert len(dets["scores"]) == 10
    assert len(dets["class_ids"]) == 10
    assert len(dets["image_ids"]) == 10
    assert dets["boxes"].shape == (10, 4)
    assert dets["scores"][0] == 0.88

    gts = data["ground_truth"]
    assert len(gts["boxes"]) == 6
    assert len(gts["class_ids"]) == 6
    assert gts["boxes"].shape == (6, 4)


def test_compute_file_sha256():
    """Test SHA256 computation on real and missing files."""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("hello vision ssd")
        temp_p = Path(f.name)

    try:
        h = compute_file_sha256(temp_p)
        assert len(h) == 64
        assert h == "acae87fe2f172e004c5385d60bc832971099dfa529ed6f46e744a73181923917"
    finally:
        temp_p.unlink(missing_ok=True)

    assert compute_file_sha256(Path("non_existent_file.csv")) == ""


def test_dump_validation_predictions_with_mock():
    """Verify dump_validation_predictions creates valid versioned npz with mock detector."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_csv = tmp_p / "test_manifest.csv"
        out_npz = tmp_p / "output_dump.npz"

        # Create dummy image file so existence check passes
        dummy_img = tmp_p / "test_img.jpg"
        dummy_img.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
        )

        # Create dummy manifest CSV
        fieldnames = [
            "image_id",
            "file_path",
            "split",
            "width",
            "height",
            "label",
            "label_id",
            "ymin",
            "xmin",
            "ymax",
            "xmax",
            "is_group_of",
            "is_occluded",
            "is_truncated",
            "is_depiction",
        ]
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "image_id": "test_001",
                    "file_path": str(dummy_img),
                    "split": "val",
                    "width": 100,
                    "height": 100,
                    "label": "Person",
                    "label_id": 1,
                    "ymin": 0.1,
                    "xmin": 0.1,
                    "ymax": 0.5,
                    "xmax": 0.5,
                    "is_group_of": 0,
                    "is_occluded": 0,
                    "is_truncated": 0,
                    "is_depiction": 0,
                }
            )

        mock_detector = MagicMock()
        mock_detector.detect.return_value = {
            "detections": [
                {
                    "class_id": 1,
                    "class_name": "Person",
                    "confidence": 0.95,
                    "box_normalized": {
                        "ymin": 0.1,
                        "xmin": 0.1,
                        "ymax": 0.5,
                        "xmax": 0.5,
                    },
                }
            ]
        }

        saved_path = dump_validation_predictions(
            model_id="fine-tuned",
            manifest_path=manifest_csv,
            output_path=out_npz,
            detector=mock_detector,
        )

        assert saved_path == out_npz
        assert out_npz.exists()

        loaded = load_predictions_dump(out_npz)
        assert loaded["metadata"]["model_id"] == "fine-tuned"
        assert loaded["metadata"]["num_detections"] == 1
        assert loaded["metadata"]["num_ground_truth"] == 1
        assert loaded["detections"]["scores"][0] == 0.95
        assert loaded["ground_truth"]["class_names"][0] == "Person"


def test_load_nonexistent_dump_raises_error():
    """Verify FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        load_predictions_dump(Path("missing_file.npz"))
