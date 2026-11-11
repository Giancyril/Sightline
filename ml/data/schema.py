"""
ml/data/schema.py
-----------------
Internal data schema and validation utilities for Open Images annotations.
Enforces normalized box coordinates [0, 1], label IDs 1..20, and flags.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ml.data.classes import CLASS_BY_ID, NUM_CLASSES


@dataclass
class NormalizedBox:
    ymin: float
    xmin: float
    ymax: float
    xmax: float

    def validate(self) -> tuple[bool, str | None]:
        """Validate box coordinates are within [0, 1] and non-degenerate."""
        if not (0.0 <= self.ymin <= 1.0 and 0.0 <= self.ymax <= 1.0):
            return False, f"y-bounds out of range: [{self.ymin}, {self.ymax}]"
        if not (0.0 <= self.xmin <= 1.0 and 0.0 <= self.xmax <= 1.0):
            return False, f"x-bounds out of range: [{self.xmin}, {self.xmax}]"
        if self.ymin >= self.ymax:
            return False, f"Degenerate height: ymin ({self.ymin}) >= ymax ({self.ymax})"
        if self.xmin >= self.xmax:
            return False, f"Degenerate width: xmin ({self.xmin}) >= xmax ({self.xmax})"
        return True, None

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class AnnotationRecord:
    image_id: str
    label: str
    label_id: int
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    is_group_of: bool = False
    is_occluded: bool = False
    is_truncated: bool = False
    is_depiction: bool = False

    def validate(self) -> tuple[bool, str | None]:
        """Validate annotation record."""
        if not (1 <= self.label_id <= NUM_CLASSES):
            return (
                False,
                f"label_id {self.label_id} out of bounds [1, {NUM_CLASSES}]",
            )
        if self.label_id in CLASS_BY_ID:
            expected_name = CLASS_BY_ID[self.label_id].name.lower()
            if self.label.strip().lower() != expected_name:
                return (
                    False,
                    f"Label name mismatch: {self.label} vs expected {expected_name}",
                )

        box = NormalizedBox(self.ymin, self.xmin, self.ymax, self.xmax)
        return box.validate()


@dataclass
class ImageRecord:
    image_id: str
    file_path: str
    width: int
    height: int
    split: str  # "train", "val", or "test"
    annotations: list[AnnotationRecord]

    def single_instance_annotations(self) -> list[AnnotationRecord]:
        """Return annotations excluding group-of clusters."""
        return [a for a in self.annotations if not a.is_group_of]

    def group_of_annotations(self) -> list[AnnotationRecord]:
        """Return annotations that are group-of clusters."""
        return [a for a in self.annotations if a.is_group_of]


MANIFEST_CSV_FIELDS = [
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


def write_manifest_csv(records: list[ImageRecord], output_path: Path | str) -> None:
    """Save image records to standard manifest CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_CSV_FIELDS)
        writer.writeheader()
        for img in records:
            if not img.annotations:
                # Background / negative image with no boxes
                writer.writerow(
                    {
                        "image_id": img.image_id,
                        "file_path": img.file_path,
                        "split": img.split,
                        "width": img.width,
                        "height": img.height,
                        "label": "__background__",
                        "label_id": 0,
                        "ymin": 0.0,
                        "xmin": 0.0,
                        "ymax": 0.0,
                        "xmax": 0.0,
                        "is_group_of": 0,
                        "is_occluded": 0,
                        "is_truncated": 0,
                        "is_depiction": 0,
                    }
                )
            else:
                for ann in img.annotations:
                    writer.writerow(
                        {
                            "image_id": img.image_id,
                            "file_path": img.file_path,
                            "split": img.split,
                            "width": img.width,
                            "height": img.height,
                            "label": ann.label,
                            "label_id": ann.label_id,
                            "ymin": round(ann.ymin, 6),
                            "xmin": round(ann.xmin, 6),
                            "ymax": round(ann.ymax, 6),
                            "xmax": round(ann.xmax, 6),
                            "is_group_of": int(ann.is_group_of),
                            "is_occluded": int(ann.is_occluded),
                            "is_truncated": int(ann.is_truncated),
                            "is_depiction": int(ann.is_depiction),
                        }
                    )


def read_manifest_csv(csv_path: Path | str) -> list[ImageRecord]:
    """Read standard manifest CSV into a list of ImageRecords."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    images_dict: dict[str, ImageRecord] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            if img_id not in images_dict:
                images_dict[img_id] = ImageRecord(
                    image_id=img_id,
                    file_path=row["file_path"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    split=row["split"],
                    annotations=[],
                )

            label_id = int(row["label_id"])
            if label_id > 0:  # Ignore background placeholder rows
                ann = AnnotationRecord(
                    image_id=img_id,
                    label=row["label"],
                    label_id=label_id,
                    ymin=float(row["ymin"]),
                    xmin=float(row["xmin"]),
                    ymax=float(row["ymax"]),
                    xmax=float(row["xmax"]),
                    is_group_of=bool(int(row.get("is_group_of", 0))),
                    is_occluded=bool(int(row.get("is_occluded", 0))),
                    is_truncated=bool(int(row.get("is_truncated", 0))),
                    is_depiction=bool(int(row.get("is_depiction", 0))),
                )
                images_dict[img_id].annotations.append(ann)

    return list(images_dict.values())
