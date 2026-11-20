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
import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.main import app
from api.routers.corrections import create_correction, delete_correction, get_correction
from api.schemas import (
    CorrectionAnnotation,
    CorrectionCreateRequest,
    ImageDimensions,
    NormalizedBBox,
)


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
