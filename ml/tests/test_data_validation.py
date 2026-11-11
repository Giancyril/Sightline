"""
ml/tests/test_data_validation.py
--------------------------------
Stage 1 Data Validation tests.
Verifies box geometry, label ID integrity, manifest round-trip,
strict split isolation (zero image overlap), and EDA report generation.
"""

from pathlib import Path

import pytest

from ml.data.classes import (
    CLASSES,
    NUM_CLASSES,
    get_class_by_id,
    get_class_by_name,
    get_coco_classes,
    get_non_coco_classes,
)
from ml.data.download import build_sample_subset
from ml.data.eda import analyze_records, generate_eda_markdown
from ml.data.schema import (
    AnnotationRecord,
    ImageRecord,
    NormalizedBox,
    read_manifest_csv,
    write_manifest_csv,
)


@pytest.mark.stage1
class TestClassDefinitions:
    def test_twenty_classes_total(self) -> None:
        assert len(CLASSES) == 20
        assert NUM_CLASSES == 20

    def test_ten_coco_and_ten_non_coco(self) -> None:
        coco = get_coco_classes()
        non_coco = get_non_coco_classes()
        assert len(coco) == 10
        assert len(non_coco) == 10

    def test_unique_ids_and_names(self) -> None:
        ids = [c.id for c in CLASSES]
        names = [c.name.lower() for c in CLASSES]
        mids = [c.mid for c in CLASSES]
        colors = [c.color_hex for c in CLASSES]

        assert len(set(ids)) == 20
        assert set(ids) == set(range(1, 21))
        assert len(set(names)) == 20
        assert len(set(mids)) == 20
        assert len(set(colors)) == 20

    def test_get_by_id_and_name(self) -> None:
        person = get_class_by_id(1)
        assert person.name == "Person"
        assert person.is_coco is True

        guitar = get_class_by_name("guitar")
        assert guitar.id == 11
        assert guitar.is_coco is False


@pytest.mark.stage1
class TestBoxValidation:
    def test_valid_box(self) -> None:
        box = NormalizedBox(0.1, 0.2, 0.8, 0.9)
        valid, msg = box.validate()
        assert valid is True
        assert msg is None
        assert box.area == pytest.approx(0.7 * 0.7)

    def test_out_of_bounds_box(self) -> None:
        box = NormalizedBox(-0.1, 0.2, 0.8, 1.2)
        valid, msg = box.validate()
        assert valid is False
        assert msg is not None

    def test_inverted_coordinates_box(self) -> None:
        box = NormalizedBox(0.8, 0.2, 0.1, 0.9)  # ymin > ymax
        valid, msg = box.validate()
        assert valid is False
        assert "Degenerate height" in str(msg)


@pytest.mark.stage1
class TestAnnotationRecord:
    def test_valid_annotation(self) -> None:
        ann = AnnotationRecord(
            image_id="img_001",
            label="Person",
            label_id=1,
            ymin=0.1,
            xmin=0.1,
            ymax=0.9,
            xmax=0.9,
        )
        valid, msg = ann.validate()
        assert valid is True
        assert msg is None

    def test_label_mismatch_fails(self) -> None:
        ann = AnnotationRecord(
            image_id="img_001",
            label="Guitar",
            label_id=1,  # ID 1 is Person, not Guitar
            ymin=0.1,
            xmin=0.1,
            ymax=0.9,
            xmax=0.9,
        )
        valid, msg = ann.validate()
        assert valid is False
        assert "mismatch" in str(msg)

    def test_group_of_filtering(self) -> None:
        ann_single = AnnotationRecord(
            "img_1", "Person", 1, 0.1, 0.1, 0.5, 0.5, is_group_of=False
        )
        ann_group = AnnotationRecord(
            "img_1", "Person", 1, 0.5, 0.5, 0.9, 0.9, is_group_of=True
        )

        img = ImageRecord(
            "img_1", "path/1.jpg", 640, 480, "train", [ann_single, ann_group]
        )
        assert len(img.single_instance_annotations()) == 1
        assert img.single_instance_annotations()[0] == ann_single
        assert len(img.group_of_annotations()) == 1
        assert img.group_of_annotations()[0] == ann_group


@pytest.mark.stage1
class TestManifestRoundtrip:
    def test_csv_manifest_write_and_read(self, tmp_path: Path) -> None:
        ann = AnnotationRecord(
            image_id="img_test_01",
            label="Laptop",
            label_id=2,
            ymin=0.2,
            xmin=0.3,
            ymax=0.6,
            xmax=0.7,
            is_group_of=False,
            is_occluded=True,
        )
        rec = ImageRecord(
            image_id="img_test_01",
            file_path="images/test.jpg",
            width=800,
            height=600,
            split="train",
            annotations=[ann],
        )

        manifest_path = tmp_path / "test_manifest.csv"
        write_manifest_csv([rec], manifest_path)
        loaded = read_manifest_csv(manifest_path)

        assert len(loaded) == 1
        assert loaded[0].image_id == "img_test_01"
        assert len(loaded[0].annotations) == 1
        loaded_ann = loaded[0].annotations[0]
        assert loaded_ann.label == "Laptop"
        assert loaded_ann.label_id == 2
        assert loaded_ann.is_occluded is True
        assert loaded_ann.ymin == pytest.approx(0.2)


@pytest.mark.stage1
class TestSubsetSplitsAndDisjointIsolation:
    def test_generate_and_validate_subset(self, tmp_path: Path) -> None:
        manifests = build_sample_subset(
            output_dir=tmp_path / "subset",
            train_count=30,
            val_count=10,
            test_count=10,
            seed=123,
        )

        train_records = read_manifest_csv(manifests["train"])
        val_records = read_manifest_csv(manifests["val"])
        test_records = read_manifest_csv(manifests["test"])

        train_ids = {r.image_id for r in train_records}
        val_ids = {r.image_id for r in val_records}
        test_ids = {r.image_id for r in test_records}

        # 1. Verify strict zero overlap between splits
        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(val_ids.intersection(test_ids)) == 0

        # 2. Verify all bounding boxes are within bounds
        for records in [train_records, val_records, test_records]:
            for img in records:
                for ann in img.annotations:
                    valid, msg = ann.validate()
                    assert valid is True, f"Failed on {img.image_id}: {msg}"

        # 3. Verify EDA report generates cleanly from stats
        stats = {
            "train": analyze_records(train_records),
            "val": analyze_records(val_records),
            "test": analyze_records(test_records),
        }
        report_path = tmp_path / "eda_test_report.md"
        generate_eda_markdown(stats, report_path)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Split Summary" in content
        assert "Class Balance" in content
