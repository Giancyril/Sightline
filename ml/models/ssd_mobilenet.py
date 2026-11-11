"""
ml/models/ssd_mobilenet.py
--------------------------
SSD MobileNet V2 model architecture, anchor decoder, and post-processor.
Implements the feature extractor, multi-scale SSD pyramid heads,
box regression decoding, and class-wise non-maximum suppression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml.utils.bbox import clip_boxes, nms


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class SSDMobileNetV2(nn.Module):
    """
    SSD MobileNet V2 single-shot detector in pure PyTorch (no torchvision required).
    Features extracted at 6 pyramid scales:
      1. MobileNetV2 stage 1 (19x19)
      2. MobileNetV2 stage 2 (10x10)
      3. Extra Conv Block 1 (5x5)
      4. Extra Conv Block 2 (3x3)
      5. Extra Conv Block 3 (2x2)
      6. Extra Conv Block 4 (1x1)
    """

    def __init__(
        self,
        num_classes: int = 81,  # 80 COCO classes + 1 background default
        num_anchors_per_scale: tuple[int, ...] = (4, 6, 6, 6, 4, 4),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_scale = num_anchors_per_scale

        # Native MobileNetV2 feature extractor
        # Input stem: (B, 3, 300, 300) -> (B, 32, 150, 150)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),  # -> 75x75
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 32, stride=2, expand_ratio=6),  # -> 38x38
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
        )

        # Stage 1: 38x38 -> 19x19 (96 channels)
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
        )

        # Stage 2: 19x19 -> 10x10 (1280 channels via Conv)
        self.stage2 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 320, stride=1, expand_ratio=6),
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        # Extra feature pyramid layers
        self.extra1 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.extra3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.extra4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.ReLU6(inplace=True),
        )

        in_channels = [96, 1280, 512, 256, 256, 256]

        # Multi-scale prediction heads
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for in_c, num_a in zip(in_channels, num_anchors_per_scale, strict=True):
            self.cls_heads.append(
                nn.Conv2d(in_c, num_a * num_classes, kernel_size=3, padding=1)
            )
            self.reg_heads.append(nn.Conv2d(in_c, num_a * 4, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, 3, 300, 300) input tensor in [-1.0, 1.0]
        Returns:
            predicted_offsets: (B, total_anchors, 4)
            class_logits: (B, total_anchors, num_classes)
        """
        s0 = self.stem(x)
        f1 = self.stage1(s0)  # (B, 96, 19, 19)
        f2 = self.stage2(f1)  # (B, 1280, 10, 10)
        f3 = self.extra1(f2)  # (B, 512, 5, 5)
        f4 = self.extra2(f3)  # (B, 256, 3, 3)
        f5 = self.extra3(f4)  # (B, 256, 2, 2)
        f6 = self.extra4(f5)  # (B, 256, 1, 1)

        feature_maps = [f1, f2, f3, f4, f5, f6]
        all_cls = []
        all_reg = []

        batch_size = x.shape[0]

        for feat, cls_head, reg_head in zip(
            feature_maps, self.cls_heads, self.reg_heads, strict=True
        ):
            c = cls_head(feat)
            c = c.permute(0, 2, 3, 1).contiguous()
            c = c.view(batch_size, -1, self.num_classes)
            all_cls.append(c)

            r = reg_head(feat)
            r = r.permute(0, 2, 3, 1).contiguous()
            r = r.view(batch_size, -1, 4)
            all_reg.append(r)

        return torch.cat(all_reg, dim=1), torch.cat(all_cls, dim=1)


def decode_box_offsets(
    predicted_offsets: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """
    Decode regression offsets [dy, dx, dh, dw] relative to prior anchors.
    Formulas:
        gy = ay + dy * ah
        gx = ax + dx * aw
        gh = ah * exp(dh)
        gw = aw * exp(dw)
    Returns:
        decoded_boxes: (N, 4) in [ymin, xmin, ymax, xmax]
    """
    ay = (anchors[:, 0] + anchors[:, 2]) / 2.0
    ax = (anchors[:, 1] + anchors[:, 3]) / 2.0
    ah = np.maximum(1e-4, anchors[:, 2] - anchors[:, 0])
    aw = np.maximum(1e-4, anchors[:, 3] - anchors[:, 1])

    dy = predicted_offsets[:, 0]
    dx = predicted_offsets[:, 1]
    dh = np.clip(predicted_offsets[:, 2], -5.0, 5.0)
    dw = np.clip(predicted_offsets[:, 3], -5.0, 5.0)

    gy = ay + dy * ah
    gx = ax + dx * aw
    gh = ah * np.exp(dh)
    gw = aw * np.exp(dw)

    ymin = gy - gh / 2.0
    xmin = gx - gw / 2.0
    ymax = gy + gh / 2.0
    xmax = gx + gw / 2.0

    boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)
    return clip_boxes(boxes)


def postprocess_ssd_predictions(
    offsets: np.ndarray,
    class_probs: np.ndarray,
    anchors: np.ndarray,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 50,
) -> list[tuple[list[float], int, float]]:
    """
    Filter and decode raw SSD predictions via per-class NMS.
    Returns:
        list of (box, class_id, score) sorted by score descending.
    """
    decoded_boxes = decode_box_offsets(offsets, anchors)
    detections: list[tuple[list[float], int, float]] = []

    num_classes = class_probs.shape[1]

    # Skip class 0 (background)
    for cls_idx in range(1, num_classes):
        scores = class_probs[:, cls_idx]
        mask = scores >= confidence_threshold
        if not np.any(mask):
            continue

        candidate_boxes = decoded_boxes[mask]
        candidate_scores = scores[mask]

        kept_indices = nms(
            candidate_boxes, candidate_scores, iou_threshold=iou_threshold
        )
        for ki in kept_indices:
            detections.append(
                (
                    candidate_boxes[ki].tolist(),
                    cls_idx,
                    float(candidate_scores[ki]),
                )
            )

    # Sort all detections across classes by score descending
    detections.sort(key=lambda d: d[2], reverse=True)
    return detections[:max_detections]
