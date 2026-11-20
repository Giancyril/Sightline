"""
api/tests/test_corrections.py
-----------------------------
Unit and integration tests for POST /v1/corrections, COCO JSON structure,
privacy default enforcement (zero image storage), opt-in image storage,
and deletion.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.main import app
from api.routers.corrections import (
    create_correction,
    delete_correction,
    export_corrections_zip,
    get_correction,
)
from api.schemas import (
    CorrectionAnnotation,
    CorrectionCreateRequest,
    ImageDimensions,
    NormalizedBBox,
)
from ml.data.schema import coco_to_image_records, read_manifest_csv, write_manifest_csv


@pytest.fixture
def tmp_corrections_dir(tmp_path: Path):
    """Fixture pointing CORRECTIONS_DIR to tmp_path."""
    orig_env = os.environ.get("CORRECTIONS_DIR")
    target = tmp_path / "corrections"
    os.environ["CORRECTIONS_DIR"] = str(target)
    yield target
    if orig_env is not None:
        os.environ["CORRECTIONS_DIR"] = orig_env
    else:
        os.environ.pop("CORRECTIONS_DIR", None)


def test_routes_registered():
    """Verify corrections routes are registered in the FastAPI app."""
    paths = list(app.openapi()["paths"].keys())
    assert "/v1/corrections" in paths
    assert "/v1/corrections/export" in paths
    assert "/v1/corrections/{correction_id}" in paths


def test_post_corrections_default_privacy_zero_image(tmp_corrections_dir: Path):
    """Test standard correction submission: stores COCO JSON, stores NO image."""
    dummy_bytes = b"fake image content for hashing"
    sha = hashlib.sha256(dummy_bytes).hexdigest()

    req = CorrectionCreateRequest(
        image_hash=sha,
        image_dimensions=ImageDimensions(width=800, height=600, channels=3),
        corrected_boxes=[
            CorrectionAnnotation(
                class_id=1,
                class_name="Person",
                box_normalized=NormalizedBBox(ymin=0.1, xmin=0.2, ymax=0.6, xmax=0.7),
                confidence=1.0,
                is_manual=True,
            ),
            CorrectionAnnotation(
                class_id=2,
                class_name="Laptop",
                box_normalized=NormalizedBBox(ymin=0.5, xmin=0.5, ymax=0.8, xmax=0.9),
                confidence=0.95,
                is_manual=False,
            ),
        ],
        original_detections=[],
        model_id="fine-tuned",
        model_version="v1.0",
        store_image=False,
    )

    res = create_correction(req)

    assert res.has_image is False
    assert res.total_annotations == 2
    assert res.image_hash == sha
    assert res.id.startswith("corr_")

    # Verify COCO file written to disk
    coco_path = tmp_corrections_dir / f"{res.id}.json"
    assert coco_path.exists()

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Check COCO format fields
    assert "info" in coco
    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco
    assert "metadata" in coco

    # Check images section
    assert len(coco["images"]) == 1
    assert coco["images"][0]["width"] == 800
    assert coco["images"][0]["height"] == 600
    assert coco["images"][0]["image_hash"] == sha
    assert coco["images"][0]["has_image"] is False

    # Check annotations section & pixel coordinates
    assert len(coco["annotations"]) == 2
    ann1 = coco["annotations"][0]
    assert ann1["category_id"] == 1
    # [xmin_px, ymin_px, width_px, height_px]
    assert ann1["bbox"] == [160.0, 60.0, 400.0, 300.0]
    assert ann1["area"] == 120000.0
    assert ann1["iscrowd"] == 0
    assert ann1["is_manual"] is True

    # Check categories has 20 classes
    assert len(coco["categories"]) == 20
    assert coco["categories"][0]["name"] == "Person"

    # Verify NO raw image is stored
    images_dir = tmp_corrections_dir / "images"
    assert not images_dir.exists() or len(list(images_dir.iterdir())) == 0


def test_post_corrections_explicit_image_opt_in(tmp_corrections_dir: Path):
    """Test user opt-in: image is safely saved when store_image=True."""
    dummy_img = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb"
    b64_img = base64.b64encode(dummy_img).decode("ascii")
    sha = hashlib.sha256(dummy_img).hexdigest()

    req = CorrectionCreateRequest(
        image_hash=sha,
        image_dimensions=ImageDimensions(width=100, height=100, channels=3),
        corrected_boxes=[
            CorrectionAnnotation(
                class_id=3,
                class_name="Chair",
                box_normalized=NormalizedBBox(ymin=0.2, xmin=0.2, ymax=0.8, xmax=0.8),
                confidence=1.0,
                is_manual=True,
            )
        ],
        original_detections=[],
        store_image=True,
        image_base64=b64_img,
    )

    res = create_correction(req)
    assert res.has_image is True

    img_dest = tmp_corrections_dir / "images" / f"{res.id}.jpg"
    assert img_dest.exists()
    assert img_dest.read_bytes() == dummy_img


def test_delete_correction(tmp_corrections_dir: Path):
    """Test deleting correction and its stored image."""
    dummy_img = b"test img"
    b64_img = base64.b64encode(dummy_img).decode("ascii")
    sha = hashlib.sha256(dummy_img).hexdigest()

    req = CorrectionCreateRequest(
        image_hash=sha,
        image_dimensions=ImageDimensions(width=100, height=100, channels=3),
        corrected_boxes=[
            CorrectionAnnotation(
                class_id=4,
                class_name="Book",
                box_normalized=NormalizedBBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
            )
        ],
        store_image=True,
        image_base64=b64_img,
    )

    create_res = create_correction(req)
    cid = create_res.id

    assert (tmp_corrections_dir / f"{cid}.json").exists()
    assert (tmp_corrections_dir / "images" / f"{cid}.jpg").exists()

    # Get
    retrieved = get_correction(cid)
    assert retrieved["metadata"]["correction_id"] == cid
    assert retrieved["images"][0]["id"] == 1

    # Delete
    del_res = delete_correction(cid)
    assert del_res["deleted"] is True

    # Verify files deleted
    assert not (tmp_corrections_dir / f"{cid}.json").exists()
    assert not (tmp_corrections_dir / "images" / f"{cid}.jpg").exists()

    # 404 on subsequent get or delete
    with pytest.raises(HTTPException) as exc_info:
        get_correction(cid)
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as exc_info:
        delete_correction(cid)
    assert exc_info.value.status_code == 404


def test_validation_errors():
    """Test payload validation (bad hash, invalid class, degenerate box)."""
    # Bad hash raises ValidationError
    with pytest.raises(ValidationError):
        CorrectionCreateRequest(
            image_hash="not-a-valid-hex-hash",
            image_dimensions=ImageDimensions(width=100, height=100),
            corrected_boxes=[],
        )

    # Invalid class ID (>20) raises ValidationError
    with pytest.raises(ValidationError):
        CorrectionAnnotation(
            class_id=99,
            class_name="Unknown",
            box_normalized=NormalizedBBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
        )

    # Degenerate box coordinates (ymin >= ymax) raises HTTPException 422 in create_correction
    degenerate_req = CorrectionCreateRequest(
        image_hash="a" * 64,
        image_dimensions=ImageDimensions(width=100, height=100),
        corrected_boxes=[
            CorrectionAnnotation(
                class_id=1,
                class_name="Person",
                box_normalized=NormalizedBBox(ymin=0.5, xmin=0.1, ymax=0.2, xmax=0.5),
            )
        ],
    )
    with pytest.raises(HTTPException) as exc_info:
        create_correction(degenerate_req)
    assert exc_info.value.status_code == 422


def test_export_corrections_zip_round_trip(tmp_corrections_dir: Path, tmp_path: Path):
    """Test exporting corrections as ZIP and round-tripping through data pipeline."""
    # 1. Create two corrections: one with store_image=False, one with store_image=True
    sha1 = hashlib.sha256(b"image 1").hexdigest()
    req1 = CorrectionCreateRequest(
        image_hash=sha1,
        image_dimensions=ImageDimensions(width=640, height=480, channels=3),
        corrected_boxes=[
            CorrectionAnnotation(
                class_id=1,
                class_name="Person",
                box_normalized=NormalizedBBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9),
                confidence=1.0,
            ),
            CorrectionAnnotation(
                class_id=3,
                class_name="Chair",
                box_normalized=NormalizedBBox(ymin=0.4, xmin=0.4, ymax=0.7, xmax=0.6),
                confidence=0.88,
            ),
        ],
        store_image=False,
    )
    c1 = create_correction(req1)

    dummy_img_bytes = b"opt-in image binary data"
    b64_img = base64.b64encode(dummy_img_bytes).decode("ascii")
    sha2 = hashlib.sha256(dummy_img_bytes).hexdigest()
    req2 = CorrectionCreateRequest(
        image_hash=sha2,
        image_dimensions=ImageDimensions(width=1024, height=768, channels=3),
        corrected_boxes=[
            CorrectionAnnotation(
                class_id=2,
                class_name="Laptop",
                box_normalized=NormalizedBBox(ymin=0.2, xmin=0.3, ymax=0.5, xmax=0.7),
                confidence=0.95,
            )
        ],
        store_image=True,
        image_base64=b64_img,
    )
    c2 = create_correction(req2)

    # 2. Call export endpoint
    export_response = export_corrections_zip(include_images=True)
    assert export_response.status_code == 200
    assert "application/zip" in export_response.media_type
    assert "corrections_coco_export.zip" in export_response.headers.get("Content-Disposition", "")

    # 3. Read ZIP file from memory
    zip_bytes = export_response.body
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "manifest.json" in namelist
        assert f"annotations/{c1.id}.json" in namelist
        assert f"annotations/{c2.id}.json" in namelist
        assert f"images/{c2.id}.jpg" in namelist

        manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest_data["total_corrections"] == 2
        assert manifest_data["total_images"] == 1
        assert manifest_data["format"] == "COCO 1.0"

        # Check raw image content
        assert zf.read(f"images/{c2.id}.jpg") == dummy_img_bytes

        # 4. Extract and round-trip COCO JSON through ml data pipeline
        coco_1_doc = json.loads(zf.read(f"annotations/{c1.id}.json").decode("utf-8"))
        records_1 = coco_to_image_records(coco_1_doc, split="train")
        assert len(records_1) == 1
        img_rec = records_1[0]
        assert img_rec.width == 640
        assert img_rec.height == 480
        assert len(img_rec.annotations) == 2

        # Validate geometric bounding box normalization
        for ann in img_rec.annotations:
            valid, msg = ann.validate()
            assert valid, f"Annotation validation failed: {msg}"

        # 5. Manifest CSV round-trip test
        manifest_csv_path = tmp_path / "corrections_manifest.csv"
        write_manifest_csv(records_1, manifest_csv_path)
        assert manifest_csv_path.exists()

        reloaded_records = read_manifest_csv(manifest_csv_path)
        assert len(reloaded_records) == 1
        reloaded = reloaded_records[0]
        assert reloaded.image_id == img_rec.image_id
        assert reloaded.width == 640
        assert reloaded.height == 480
        assert len(reloaded.annotations) == 2
        assert reloaded.annotations[0].label == "Person"
        assert reloaded.annotations[0].label_id == 1
        assert abs(reloaded.annotations[0].ymin - 0.1) < 1e-4
        assert abs(reloaded.annotations[0].xmin - 0.2) < 1e-4
        assert reloaded.annotations[1].label == "Chair"
        assert reloaded.annotations[1].label_id == 3

