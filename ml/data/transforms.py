"""
ml/data/transforms.py
---------------------
SSD Data Preprocessing, Augmentations, Anchor Generation, and Target Encoding.
Includes horizontal flip, random crop with box clipping, color jitter,
resize/pad to 300x300, normalization, and SSD target encoding.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageEnhance

from ml.utils.bbox import clip_boxes, iou


def resize_or_pad(
    image: Image.Image | np.ndarray,
    boxes: np.ndarray,
    target_size: tuple[int, int] = (300, 300),
    preserve_aspect_ratio: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize or pad image to target SSD dimensions (default 300x300).
    Adjusts normalized box coordinates [ymin, xmin, ymax, xmax].
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image

    orig_w, orig_h = pil_img.size
    target_w, target_h = target_size

    if not preserve_aspect_ratio:
        # Standard SSD direct resize
        resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_arr = np.array(resized, dtype=np.float32)
        # Normalized coordinates remain identical under direct linear resize
        return img_arr, boxes.copy()

    # Aspect-ratio preserving resize with zero-padding
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    padded = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))

    # Transform normalized boxes to padded canvas normalized coords
    adj_boxes = boxes.copy()
    if len(boxes) > 0:
        adj_boxes[:, 0] = (boxes[:, 0] * new_h + pad_y) / target_h
        adj_boxes[:, 1] = (boxes[:, 1] * new_w + pad_x) / target_w
        adj_boxes[:, 2] = (boxes[:, 2] * new_h + pad_y) / target_h
        adj_boxes[:, 3] = (boxes[:, 3] * new_w + pad_x) / target_w

    return np.array(padded, dtype=np.float32), adj_boxes


def random_horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
    prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random horizontal flip for image and normalized bounding boxes.
    Box transform: xmin' = 1 - xmax, xmax' = 1 - xmin.
    """
    if random.random() >= prob:
        return image, boxes

    flipped_img = np.fliplr(image).copy()
    flipped_boxes = boxes.copy()
    if len(boxes) > 0:
        xmin = boxes[:, 1].copy()
        xmax = boxes[:, 3].copy()
        flipped_boxes[:, 1] = 1.0 - xmax
        flipped_boxes[:, 3] = 1.0 - xmin

    return flipped_img, flipped_boxes


def random_crop_with_box_clipping(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    min_iou: float = 0.3,
    prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random crop patch with box clipping and box center preservation.
    Discards boxes whose centers fall outside the crop or IoU < min_iou.
    """
    if random.random() >= prob or len(boxes) == 0:
        return image, boxes, labels

    h, w = image.shape[:2]

    for _ in range(10):  # Try up to 10 crop attempts
        crop_w = random.uniform(0.6, 1.0)
        crop_h = random.uniform(0.6, 1.0)
        crop_xmin = random.uniform(0.0, 1.0 - crop_w)
        crop_ymin = random.uniform(0.0, 1.0 - crop_h)
        crop_xmax = crop_xmin + crop_w
        crop_ymax = crop_ymin + crop_h

        # Check box center containment
        box_centers_y = (boxes[:, 0] + boxes[:, 2]) / 2.0
        box_centers_x = (boxes[:, 1] + boxes[:, 3]) / 2.0
        inside_mask = (
            (box_centers_y >= crop_ymin)
            & (box_centers_y <= crop_ymax)
            & (box_centers_x >= crop_xmin)
            & (box_centers_x <= crop_xmax)
        )

        if not np.any(inside_mask):
            continue

        selected_boxes = boxes[inside_mask]
        selected_labels = labels[inside_mask]

        # Clip boxes to crop patch
        clipped_boxes = selected_boxes.copy()
        clipped_boxes[:, 0] = np.maximum(clipped_boxes[:, 0], crop_ymin)
        clipped_boxes[:, 1] = np.maximum(clipped_boxes[:, 1], crop_xmin)
        clipped_boxes[:, 2] = np.minimum(clipped_boxes[:, 2], crop_ymax)
        clipped_boxes[:, 3] = np.minimum(clipped_boxes[:, 3], crop_xmax)

        # Discard degenerate clipped boxes
        valid_boxes = (clipped_boxes[:, 2] > clipped_boxes[:, 0] + 0.01) & (
            clipped_boxes[:, 3] > clipped_boxes[:, 1] + 0.01
        )
        if not np.any(valid_boxes):
            continue

        clipped_boxes = clipped_boxes[valid_boxes]
        selected_labels = selected_labels[valid_boxes]

        # Rescale box coordinates to [0, 1] relative to the crop
        rel_boxes = clipped_boxes.copy()
        rel_boxes[:, 0] = (clipped_boxes[:, 0] - crop_ymin) / crop_h
        rel_boxes[:, 1] = (clipped_boxes[:, 1] - crop_xmin) / crop_w
        rel_boxes[:, 2] = (clipped_boxes[:, 2] - crop_ymin) / crop_h
        rel_boxes[:, 3] = (clipped_boxes[:, 3] - crop_xmin) / crop_w

        # Slice image patch
        py1, py2 = int(crop_ymin * h), int(crop_ymax * h)
        px1, px2 = int(crop_xmin * w), int(crop_xmax * w)
        cropped_img = image[py1:py2, px1:px2]

        return cropped_img, clip_boxes(rel_boxes), selected_labels

    return image, boxes, labels


def color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    """Photometric distortions (brightness, contrast, saturation)."""
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))

    if brightness > 0:
        factor = random.uniform(1.0 - brightness, 1.0 + brightness)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    if contrast > 0:
        factor = random.uniform(1.0 - contrast, 1.0 + contrast)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

    if saturation > 0:
        factor = random.uniform(1.0 - saturation, 1.0 + saturation)
        pil_img = ImageEnhance.Color(pil_img).enhance(factor)

    return np.array(pil_img, dtype=np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize uint8 [0, 255] image to [-1.0, 1.0] for SSD MobileNet."""
    return (image / 127.5) - 1.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize [-1.0, 1.0] to uint8 [0, 255]."""
    return np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# SSD Anchor Generation & Target Encoding
# ---------------------------------------------------------------------------


def generate_ssd_anchors(
    feature_map_sizes: Sequence[int] = (19, 10, 5, 3, 2, 1),
    min_scale: float = 0.2,
    max_scale: float = 0.9,
    aspect_ratios: Sequence[Sequence[float]] = (
        (1.0, 2.0, 0.5),
        (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
        (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
        (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
        (1.0, 2.0, 0.5),
        (1.0, 2.0, 0.5),
    ),
) -> np.ndarray:
    """
    Generate default prior anchor boxes for SSD300.
    Returns: (N_anchors, 4) in [ymin, xmin, ymax, xmax] normalized coordinates.
    """
    num_layers = len(feature_map_sizes)
    scales = np.linspace(min_scale, max_scale, num_layers)
    anchors: list[list[float]] = []

    for k, f_k in enumerate(feature_map_sizes):
        s_k = scales[k]
        s_k_next = (
            scales[k + 1] if k + 1 < num_layers else np.sqrt(s_k * (max_scale + 0.1))
        )
        ratios = aspect_ratios[k]

        for i in range(f_k):
            for j in range(f_k):
                cy = (i + 0.5) / f_k
                cx = (j + 0.5) / f_k

                # Ratio 1.0 default
                anchors.append([cy - s_k / 2, cx - s_k / 2, cy + s_k / 2, cx + s_k / 2])
                # Ratio 1.0 with geometric mean scale
                s_prime = np.sqrt(s_k * s_k_next)
                anchors.append(
                    [
                        cy - s_prime / 2,
                        cx - s_prime / 2,
                        cy + s_prime / 2,
                        cx + s_prime / 2,
                    ]
                )

                for r in ratios:
                    if r == 1.0:
                        continue
                    w = s_k * np.sqrt(r)
                    h = s_k / np.sqrt(r)
                    anchors.append([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2])

    return clip_boxes(np.array(anchors, dtype=np.float32))


def encode_targets(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    anchors: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match ground-truth boxes to anchor priors for SSD loss.

    Args:
        gt_boxes: (N, 4) [ymin, xmin, ymax, xmax]
        gt_labels: (N,) label IDs [1..20]
        anchors: (M, 4) anchor boxes [ymin, xmin, ymax, xmax]
        iou_threshold: IoU overlap to consider an anchor positive.

    Returns:
        target_offsets: (M, 4) [dy, dx, dh, dw]
        target_classes: (M,) 0 for background, 1..20 for matched class
    """
    num_anchors = len(anchors)
    target_classes = np.zeros((num_anchors,), dtype=np.int64)
    target_offsets = np.zeros((num_anchors, 4), dtype=np.float32)

    if len(gt_boxes) == 0:
        return target_offsets, target_classes

    # Compute IoU matrix (M, N)
    ious = np.zeros((num_anchors, len(gt_boxes)), dtype=np.float32)
    for a_idx, a_box in enumerate(anchors):
        for g_idx, g_box in enumerate(gt_boxes):
            ious[a_idx, g_idx] = iou(a_box, g_box)

    # 1. Best anchor for each ground truth (guarantees every GT has at least 1 anchor)
    best_anchor_for_gt = np.argmax(ious, axis=0)
    for g_idx, a_idx in enumerate(best_anchor_for_gt):
        ious[a_idx, g_idx] = 2.0  # Force match

    # 2. For each anchor, match best GT if IoU >= threshold
    best_gt_for_anchor = np.argmax(ious, axis=1)
    max_iou_for_anchor = np.max(ious, axis=1)

    matched_mask = max_iou_for_anchor >= iou_threshold
    target_classes[matched_mask] = gt_labels[best_gt_for_anchor[matched_mask]]

    # 3. Compute regression offsets: [dy, dx, dh, dw]
    matched_gt = gt_boxes[best_gt_for_anchor]
    ay = (anchors[:, 0] + anchors[:, 2]) / 2.0
    ax = (anchors[:, 1] + anchors[:, 3]) / 2.0
    ah = np.maximum(1e-4, anchors[:, 2] - anchors[:, 0])
    aw = np.maximum(1e-4, anchors[:, 3] - anchors[:, 1])

    gy = (matched_gt[:, 0] + matched_gt[:, 2]) / 2.0
    gx = (matched_gt[:, 1] + matched_gt[:, 3]) / 2.0
    gh = np.maximum(1e-4, matched_gt[:, 2] - matched_gt[:, 0])
    gw = np.maximum(1e-4, matched_gt[:, 3] - matched_gt[:, 1])

    target_offsets[:, 0] = (gy - ay) / ah
    target_offsets[:, 1] = (gx - ax) / aw
    target_offsets[:, 2] = np.log(gh / ah)
    target_offsets[:, 3] = np.log(gw / aw)

    return target_offsets, target_classes
