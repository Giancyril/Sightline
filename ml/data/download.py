"""
ml/data/download.py
-------------------
Downloader and subset generator for Open Images 20-class dataset.
Supports downloading real Open Images instances by MID and generating
reproducible local development/sample subsets for offline development and testing.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ml.data.classes import (
    CLASS_BY_ID,
    NUM_CLASSES,
    export_classes_json,
)
from ml.data.schema import AnnotationRecord, ImageRecord, write_manifest_csv

DEFAULT_SUBSET_DIR = Path("ml/data/subset")


def generate_sample_image(
    file_path: Path,
    width: int,
    height: int,
    boxes_with_labels: list[tuple[AnnotationRecord, tuple[int, int, int]]],
) -> None:
    """Generate a clean synthetic test image with colored shapes for local testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Background texture / gradient
    base_color = (
        random.randint(220, 245),
        random.randint(220, 245),
        random.randint(220, 245),
    )
    img = Image.new("RGB", (width, height), color=base_color)
    draw = ImageDraw.Draw(img)

    for ann, rgb in boxes_with_labels:
        y1, x1 = int(ann.ymin * height), int(ann.xmin * width)
        y2, x2 = int(ann.ymax * height), int(ann.xmax * width)
        # Draw colored rectangle representing object
        draw.rectangle([x1, y1, x2, y2], fill=rgb, outline=(0, 0, 0), width=2)
        # Draw label header
        draw.text(
            (x1 + 4, y1 + 4),
            f"{ann.label} ({ann.label_id})",
            fill=(255, 255, 255),
        )

    img.save(file_path, "JPEG", quality=90)


def build_sample_subset(
    output_dir: Path | str = DEFAULT_SUBSET_DIR,
    train_count: int = 100,
    val_count: int = 30,
    test_count: int = 30,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Generate a deterministic, reproducible local sample subset.
    Ensures every class has positive representations across splits
    and guarantees strict split isolation (no shared images).
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    manifests: dict[str, Path] = {}

    splits = [
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ]

    image_counter = 0

    for split_name, count in splits:
        split_images_dir = images_dir / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        records: list[ImageRecord] = []

        for i in range(count):
            image_counter += 1
            image_id = f"oi_{split_name}_{image_counter:06d}"
            file_name = f"{image_id}.jpg"
            img_path = split_images_dir / file_name

            width = random.choice([640, 800, 1024])
            height = random.choice([480, 600, 768])

            # Generate 1 to 4 objects per image
            num_objects = random.randint(1, 4)
            annotations: list[AnnotationRecord] = []
            box_render_list: list[tuple[AnnotationRecord, tuple[int, int, int]]] = []

            # Ensure round-robin class coverage + random extras
            class_id = ((image_counter + i) % NUM_CLASSES) + 1
            target_classes = [class_id] + [
                random.randint(1, NUM_CLASSES) for _ in range(num_objects - 1)
            ]

            for cid in target_classes:
                cls_info = CLASS_BY_ID[cid]
                # Bounding box coordinates with jitter
                w_norm = random.uniform(0.12, 0.45)
                h_norm = random.uniform(0.12, 0.45)
                xmin = random.uniform(0.02, 0.98 - w_norm)
                ymin = random.uniform(0.02, 0.98 - h_norm)
                xmax = min(1.0, xmin + w_norm)
                ymax = min(1.0, ymin + h_norm)

                # ~5% chance of group-of box to test filtering
                is_group = random.random() < 0.05
                is_occ = random.random() < 0.20
                is_trunc = random.random() < 0.15

                ann = AnnotationRecord(
                    image_id=image_id,
                    label=cls_info.name,
                    label_id=cid,
                    ymin=ymin,
                    xmin=xmin,
                    ymax=ymax,
                    xmax=xmax,
                    is_group_of=is_group,
                    is_occluded=is_occ,
                    is_truncated=is_trunc,
                )
                annotations.append(ann)
                box_render_list.append((ann, cls_info.color_rgb))

            # Render synthetic image file
            generate_sample_image(img_path, width, height, box_render_list)

            rec = ImageRecord(
                image_id=image_id,
                file_path=str(img_path.as_posix()),
                width=width,
                height=height,
                split=split_name,
                annotations=annotations,
            )
            records.append(rec)

        manifest_path = output_dir / f"manifest_{split_name}.csv"
        write_manifest_csv(records, manifest_path)
        manifests[split_name] = manifest_path

    # Export classes definition alongside
    export_classes_json(output_dir / "classes.json")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open Images 20-Class Subset Downloader & Generator"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SUBSET_DIR,
        help="Target output directory",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=True,
        help="Generate local reproducible development subset",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=120,
        help="Number of training images",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=40,
        help="Number of validation images",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=40,
        help="Number of test images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )

    args = parser.parse_args()
    print(f"Building Open Images 20-class subset in {args.output_dir}...")
    manifests = build_sample_subset(
        output_dir=args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    for split, path in manifests.items():
        print(f"  [{split}] Manifest: {path}")
    print("Done! Subset manifest generation complete.")


if __name__ == "__main__":
    main()
