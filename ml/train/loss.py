"""
ml/train/loss.py
----------------
SSD MultiBox Loss function.
Combines Smooth L1 localization loss on positive anchors with
Hard Negative Mining (3:1 negative to positive ratio) cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    """
    SSD MultiBox Loss function.

    Args:
        neg_pos_ratio: Ratio of negative to positive anchor samples (default 3.0).
        alpha: Weight multiplier for localization loss (default 1.0).
        smooth_l1_beta: Huber loss threshold (default 1.0).
    """

    def __init__(
        self,
        neg_pos_ratio: float = 3.0,
        alpha: float = 1.0,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1_beta = smooth_l1_beta

    def forward(
        self,
        predicted_offsets: torch.Tensor,  # (B, N_anchors, 4)
        class_logits: torch.Tensor,  # (B, N_anchors, num_classes)
        target_offsets: torch.Tensor,  # (B, N_anchors, 4)
        target_classes: torch.Tensor,  # (B, N_anchors)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SSD MultiBox Loss.
        Returns:
            total_loss: combined scalar loss
            loc_loss: regression loss
            cls_loss: classification loss with hard negative mining
        """
        pos_mask = target_classes > 0  # (B, N_anchors)
        num_pos = pos_mask.sum(dim=1, keepdim=True)  # (B, 1)

        # 1. Localization Loss (Smooth L1 over positive anchors only)
        pos_idx = pos_mask.unsqueeze(-1).expand_as(predicted_offsets)
        pos_pred_offsets = predicted_offsets[pos_idx].view(-1, 4)
        pos_target_offsets = target_offsets[pos_idx].view(-1, 4)

        if pos_pred_offsets.numel() > 0:
            loc_loss = F.smooth_l1_loss(
                pos_pred_offsets,
                pos_target_offsets,
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
        else:
            loc_loss = torch.tensor(0.0, device=predicted_offsets.device)

        # 2. Classification Loss with Hard Negative Mining
        batch_size, num_anchors, num_classes = class_logits.shape
        # Compute cross-entropy for all anchors
        ce_loss = F.cross_entropy(
            class_logits.view(-1, num_classes),
            target_classes.view(-1),
            reduction="none",
        ).view(batch_size, num_anchors)

        # Filter positive anchors loss
        pos_cls_loss = ce_loss * pos_mask.float()

        # Hard negative mining: sort negative anchor losses descending
        neg_ce_loss = ce_loss * (~pos_mask).float()
        # Rank negatives
        _, neg_order = neg_ce_loss.sort(dim=1, descending=True)
        _, neg_rank = neg_order.sort(dim=1)

        # Determine number of negatives to sample: clamp to at least 1
        num_neg = torch.clamp(
            num_pos * self.neg_pos_ratio,
            min=1,
            max=num_anchors - 1,
        )
        neg_mask = neg_rank < num_neg  # (B, N_anchors)
        neg_cls_loss = ce_loss * neg_mask.float()

        # Total classification loss
        cls_loss = pos_cls_loss.sum() + neg_cls_loss.sum()

        # Normalize by total positive anchors across batch
        total_pos = max(1.0, float(pos_mask.sum().item()))
        loc_loss = loc_loss / total_pos
        cls_loss = cls_loss / total_pos

        total_loss = cls_loss + self.alpha * loc_loss
        return total_loss, loc_loss, cls_loss
