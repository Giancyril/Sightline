"""
ml/tests/test_pipeline.py
-------------------------
Stage 2 Input Pipeline tests.
Verifies data transforms, bounding-box math under augmentations,
SSD anchor generation, target encoding, TFRecord serialization,
visual contact sheet generation, and throughput measurement.
"""

from pathlib import Path

import numpy as np
import pytest

from ml.data.pipeline import DetectionDataset, benchmark_pipeline_throughput
from ml.data.tfrecords import count_tfrecord_records, create_tfrecords
from ml.data.transforms import (
    denormalize_image,
    encode_targets,
    generate_ssd_anchors,
    normalize_image,
    random_crop_with_box_clipping,
    random_horizontal_flip,
    resize_or_pad,
)
from ml.utils.drawing import create_contact_sheet


@pytest.mark.stage2
class TestDataTransforms:
    def test_horizontal_flip_coordinates(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.float32)
        boxes = np.array([[0.1, 0.2, 0.5, 0.6]], dtype=np.float32)

        flipped_img, flipped_boxes = random_horizontal_flip(img, boxes, prob=1.0)

        # y coordinates unchanged
        assert flipped_boxes[0, 0] == pytest.approx(0.1)
        assert flipped_boxes[0, 2] == pytest.approx(0.5)
        # x coordinates inverted: xmin' = 1 - 0.6 = 0.4, xmax' = 1 - 0.2 = 0.8
        assert flipped_boxes[0, 1] == pytest.approx(0.4)
        assert flipped_boxes[0, 3] == pytest.approx(0.8)

    def test_resize_or_pad(self) -> None:
        img = np.ones((200, 400, 3), dtype=np.float32) * 200
        boxes = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)

        # 1. Direct resize
        direct_img, direct_boxes = resize_or_pad(
            img, boxes, target_size=(300, 300), preserve_aspect_ratio=False
        )
        assert direct_img.shape == (300, 300, 3)
        assert np.allclose(direct_boxes, boxes)

        # 2. Padded resize
        padded_img, padded_boxes = resize_or_pad(
            img, boxes, target_size=(300, 300), preserve_aspect_ratio=True
        )
        assert padded_img.shape == (300, 300, 3)
        assert 0.0 <= padded_boxes[0, 0] <= padded_boxes[0, 2] <= 1.0
        assert 0.0 <= padded_boxes[0, 1] <= padded_boxes[0, 3] <= 1.0

    def test_crop_with_box_clipping(self) -> None:
        img = np.ones((300, 300, 3), dtype=np.float32) * 150
        boxes = np.array([[0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)

        cropped_img, clipped_boxes, clipped_labels = random_crop_with_box_clipping(
            img, boxes, labels, prob=1.0
        )
        assert cropped_img.ndim == 3
        if len(clipped_boxes) > 0:
            assert np.all(clipped_boxes >= 0.0)
            assert np.all(clipped_boxes <= 1.0)
            assert len(clipped_labels) == len(clipped_boxes)

    def test_normalization_roundtrip(self) -> None:
        img = np.random.randint(0, 256, (50, 50, 3)).astype(np.float32)
        norm = normalize_image(img)
        assert -1.0 <= norm.min() and norm.max() <= 1.0
        denorm = denormalize_image(norm)
        assert np.max(np.abs(denorm - img)) <= 1.0


@pytest.mark.stage2
class TestSSDAnchorsAndTargetEncoding:
    def test_ssd_anchors_shape(self) -> None:
        anchors = generate_ssd_anchors()
        assert anchors.ndim == 2
        assert anchors.shape[1] == 4
        assert len(anchors) > 1000  # SSD300 typically has ~1917 or ~3000 anchors
        assert np.all(anchors >= 0.0)
        assert np.all(anchors <= 1.0)

    def test_encode_targets_matching(self) -> None:
        anchors = generate_ssd_anchors()
        gt_boxes = np.array(
            [[0.2, 0.2, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]], dtype=np.float32
        )
        gt_labels = np.array([1, 11], dtype=np.int64)  # Person, Guitar

        offsets, classes = encode_targets(
            gt_boxes, gt_labels, anchors, iou_threshold=0.5
        )

        assert offsets.shape == (len(anchors), 4)
        assert classes.shape == (len(anchors),)
        # At least the best matching anchor for each GT must be positive
        assert 1 in classes
        assert 11 in classes
        # The majority of anchors should remain background (0)
        assert np.sum(classes == 0) > np.sum(classes > 0)


@pytest.mark.stage2
class TestTFRecordPipeline:
    def test_create_and_count_tfrecords(self, tmp_path: Path) -> None:
        manifest_path = Path("ml/data/subset/manifest_val.csv")
        tfrecord_path = tmp_path / "val.tfrecord"

        count = create_tfrecords(manifest_path, tfrecord_path)
        assert count == 40
        assert tfrecord_path.exists()
        assert tfrecord_path.stat().st_size > 0

        verified_count = count_tfrecord_records(tfrecord_path)
        assert verified_count == 40


@pytest.mark.stage2
class TestVisualContactSheetAndThroughput:
    def test_contact_sheet_generation(self) -> None:
        manifest_path = Path("ml/data/subset/manifest_train.csv")
        dataset = DetectionDataset(manifest_path, is_training=True, augment=True)

        samples = []
        for i in range(6):  # 2x3 grid
            img_norm, _, _ = dataset.get_sample(i)
            img_uint8 = denormalize_image(img_norm)
            rec = dataset.records[i]
            boxes = [
                [a.ymin, a.xmin, a.ymax, a.xmax]
                for a in rec.single_instance_annotations()
            ]
            labels = [a.label_id for a in rec.single_instance_annotations()]
            samples.append((img_uint8, boxes, labels))

        out_path = Path("docs/contact_sheet.png")
        sheet = create_contact_sheet(
            samples, grid_cols=3, cell_size=(300, 300), output_path=out_path
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 1000
        assert sheet.size[0] > 600

    def test_pipeline_throughput(self) -> None:
        manifest_path = Path("ml/data/subset/manifest_train.csv")
        stats = benchmark_pipeline_throughput(
            manifest_path, batch_size=16, num_batches=3
        )
        assert stats["images_processed"] > 0
        assert stats["images_per_second"] > 0.0
        assert stats["ms_per_batch"] > 0.0
        print(
            f"\nThroughput: {stats['images_per_second']:.1f} img/s ({stats['ms_per_batch']:.1f} ms/batch)"
        )
