"""
ml/utils/drawing.py
-------------------
Visualization and contact sheet utilities for bounding boxes and detections.
Uses OpenCV when available, with a full PIL/Pillow implementation so visualization
always works seamlessly across any Python environment.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ml.data.classes import CLASS_BY_ID

HAS_OPENCV = importlib.util.find_spec("cv2") is not None


def draw_bounding_boxes(
    image: Image.Image | np.ndarray,
    boxes: Sequence[Sequence[float]],  # [[ymin, xmin, ymax, xmax], ...]
    label_ids: Sequence[int],
    scores: Sequence[float] | None = None,
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw colored bounding boxes with labels and optional scores on an image.
    Coordinates are normalized [ymin, xmin, ymax, xmax] in [0, 1].
    """
    if isinstance(image, np.ndarray):
        if image.dtype in (np.float32, np.float64):
            # Scale [0, 1] or [-1, 1] to uint8 [0, 255]
            if image.min() < 0:
                image = ((image + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image).copy()
    else:
        pil_img = image.copy()

    width, height = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    for idx, (box, label_id) in enumerate(zip(boxes, label_ids, strict=False)):
        ymin, xmin, ymax, xmax = box
        px_ymin = int(round(ymin * height))
        px_xmin = int(round(xmin * width))
        px_ymax = int(round(ymax * height))
        px_xmax = int(round(xmax * width))

        # Clamp to image bounds
        px_ymin = max(0, min(height - 1, px_ymin))
        px_xmin = max(0, min(width - 1, px_xmin))
        px_ymax = max(0, min(height - 1, px_ymax))
        px_xmax = max(0, min(width - 1, px_xmax))

        cls_info = CLASS_BY_ID.get(label_id)
        color = cls_info.color_rgb if cls_info else (0, 107, 255)
        label_text = cls_info.name if cls_info else f"Class {label_id}"

        if scores is not None and idx < len(scores):
            label_text = f"{label_text} {scores[idx]:.2f}"

        # Draw box outline
        for offset in range(line_width):
            draw.rectangle(
                [
                    px_xmin - offset,
                    px_ymin - offset,
                    px_xmax + offset,
                    px_ymax + offset,
                ],
                outline=color,
            )

        # Draw chip banner
        text_bbox = draw.textbbox((px_xmin, px_ymin), label_text)
        tw = text_bbox[2] - text_bbox[0] + 6
        th = text_bbox[3] - text_bbox[1] + 4
        chip_y = max(0, px_ymin - th)
        draw.rectangle([px_xmin, chip_y, px_xmin + tw, chip_y + th], fill=color)
        draw.text((px_xmin + 3, chip_y + 2), label_text, fill=(255, 255, 255))

    return pil_img


def create_contact_sheet(
    images_with_boxes: list[
        tuple[Image.Image | np.ndarray, Sequence[Sequence[float]], Sequence[int]]
    ],
    grid_cols: int = 3,
    cell_size: tuple[int, int] = (300, 300),
    padding: int = 8,
    bg_color: tuple[int, int, int] = (245, 245, 245),
    output_path: Path | str | None = None,
) -> Image.Image:
    """
    Renders an N x M contact sheet of annotated images.
    Saves to output_path if provided.
    """
    num_images = len(images_with_boxes)
    if num_images == 0:
        return Image.new("RGB", cell_size, color=bg_color)

    cols = min(grid_cols, num_images)
    rows = (num_images + cols - 1) // cols

    cell_w, cell_h = cell_size
    sheet_w = cols * cell_w + (cols + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg_color)

    for i, (img, boxes, labels) in enumerate(images_with_boxes):
        annotated = draw_bounding_boxes(img, boxes, labels)
        resized = annotated.resize(cell_size, Image.Resampling.BILINEAR)

        c = i % cols
        r = i // cols
        x = padding + c * (cell_w + padding)
        y = padding + r * (cell_h + padding)
        sheet.paste(resized, (x, y))

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, "PNG")

    return sheet
