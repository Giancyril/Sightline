"""
ml/data/pipeline.py
-------------------
Data pipeline for SSD model training and evaluation.
Provides batching, shuffling, multi-stage augmentation, anchor target encoding,
and pipeline throughput benchmarking.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
from PIL import Image

from ml.data.schema import read_manifest_csv
from ml.data.transforms import (
    color_jitter,
    encode_targets,
    generate_ssd_anchors,
    normalize_image,
    random_crop_with_box_clipping,
    random_horizontal_flip,
    resize_or_pad,
)


class DetectionDataset:
    """Dataset providing transformed and augmented image samples and target anchors."""

    def __init__(
        self,
        manifest_path: Path | str,
        is_training: bool = True,
        target_size: tuple[int, int] = (300, 300),
        anchors: np.ndarray | None = None,
        augment: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.is_training = is_training
        self.target_size = target_size
        self.augment = augment and is_training
        self.records = read_manifest_csv(self.manifest_path)
        self.anchors = anchors if anchors is not None else generate_ssd_anchors()

    def __len__(self) -> int:
        return len(self.records)

    def get_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads and augments a single sample.
        Returns:
            image: (300, 300, 3) float32 in [-1.0, 1.0]
            target_offsets: (N_anchors, 4) float32
            target_classes: (N_anchors,) int64
        """
        record = self.records[index % len(self.records)]
        img_path = Path(record.file_path)

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (record.width, record.height), (128, 128, 128))

        img_np = np.array(img, dtype=np.float32)

        # Filter out group-of annotations for training
        single_instances = record.single_instance_annotations()
        if single_instances:
            boxes = np.array(
                [[a.ymin, a.xmin, a.ymax, a.xmax] for a in single_instances],
                dtype=np.float32,
            )
            labels = np.array([a.label_id for a in single_instances], dtype=np.int64)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        # Apply training augmentations
        if self.augment:
            # 1. Random crop with box clipping
            img_np, boxes, labels = random_crop_with_box_clipping(
                img_np, boxes, labels, min_iou=0.3, prob=0.5
            )
            # 2. Random horizontal flip
            img_np, boxes = random_horizontal_flip(img_np, boxes, prob=0.5)
            # 3. Color jitter
            img_np = color_jitter(img_np)

        # Resize to SSD target size
        img_np, boxes = resize_or_pad(img_np, boxes, target_size=self.target_size)

        # Normalize to [-1.0, 1.0]
        norm_img = normalize_image(img_np)

        # Encode targets against SSD anchor priors
        target_offsets, target_classes = encode_targets(
            boxes, labels, self.anchors, iou_threshold=0.5
        )

        return norm_img, target_offsets, target_classes

    def batch_iterator(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        drop_remainder: bool = False,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generator yielding batched tensors."""
        indices = list(range(len(self.records)))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            if drop_remainder and len(batch_idx) < batch_size:
                break

            batch_imgs = []
            batch_offsets = []
            batch_classes = []

            for idx in batch_idx:
                img, offsets, classes = self.get_sample(idx)
                batch_imgs.append(img)
                batch_offsets.append(offsets)
                batch_classes.append(classes)

            yield (
                np.stack(batch_imgs, axis=0),
                np.stack(batch_offsets, axis=0),
                np.stack(batch_classes, axis=0),
            )


def benchmark_pipeline_throughput(
    manifest_path: Path | str,
    batch_size: int = 16,
    num_batches: int = 5,
) -> dict[str, float]:
    """
    Benchmarks input pipeline throughput (images per second and latency).
    """
    dataset = DetectionDataset(manifest_path, is_training=True, augment=True)
    iterator = dataset.batch_iterator(batch_size=batch_size, shuffle=True)

    # Warm-up single batch
    _ = next(iterator, None)

    start_time = time.perf_counter()
    batches_processed = 0
    images_processed = 0

    for _ in range(num_batches):
        batch = next(iterator, None)
        if batch is None:
            break
        batches_processed += 1
        images_processed += len(batch[0])

    elapsed = max(1e-4, time.perf_counter() - start_time)
    fps = images_processed / elapsed
    ms_per_batch = (elapsed / max(1, batches_processed)) * 1000.0

    stats = {
        "images_processed": float(images_processed),
        "batches_processed": float(batches_processed),
        "elapsed_seconds": elapsed,
        "images_per_second": fps,
        "ms_per_batch": ms_per_batch,
    }
    return stats
