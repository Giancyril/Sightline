"""
ml/tests/test_fine_tuning.py
----------------------------
Stage 4 unit tests: fine-tuning pipeline, MultiBoxLoss, cosine LR,
and training artefact validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from ml.train.loss import MultiBoxLoss
from ml.train.train import (
    build_fine_tune_model,
    get_cosine_lr,
    unfreeze_top_layers,
)

pytestmark = pytest.mark.stage4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 21
NUM_ANCHORS = 64  # small synthetic anchor count for tests


@pytest.fixture()
def model():
    """Return a fresh fine-tune model with frozen backbone."""
    return build_fine_tune_model(num_classes=NUM_CLASSES, freeze_backbone=True)


@pytest.fixture()
def loss_fn():
    return MultiBoxLoss(neg_pos_ratio=3.0, alpha=1.0, smooth_l1_beta=1.0)


# ---------------------------------------------------------------------------
# build_fine_tune_model
# ---------------------------------------------------------------------------


def test_build_fine_tune_model_frozen_backbone(model):
    """Backbone (stem, stage1, stage2) must be frozen; head must be trainable."""
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    # At least some backbone params frozen
    assert len(frozen_params) > 0, "Expected backbone parameters to be frozen"

    # Detection head (cls_heads, reg_heads) must be trainable
    head_trainable = [
        p for p in trainable_params if "cls_heads" in p or "reg_heads" in p
    ]
    assert len(head_trainable) > 0, "Detection head parameters must be trainable"

    # Stem and stage1 must be frozen
    stem_frozen = [
        p for p in frozen_params if p.startswith("stem.") or p.startswith("stage1.")
    ]
    assert len(stem_frozen) > 0, "Stem and stage1 must be frozen"


def test_build_fine_tune_model_num_classes(model):
    """Model must output correct number of classes."""
    # cls_convs predict num_classes channels per anchor
    # Spot-check by running a small forward pass
    x = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        locs, scores = model(x)
    # scores shape: (1, total_anchors, num_classes)
    assert scores.shape[-1] == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} class channels, got {scores.shape[-1]}"
    )
    assert locs.shape[-1] == 4, "Localization output must have 4 coords per anchor"


# ---------------------------------------------------------------------------
# unfreeze_top_layers
# ---------------------------------------------------------------------------


def test_unfreeze_top_layers(model):
    """After unfreezing, stage2 / extra1 / extra2 params must be trainable."""
    # Confirm stage2 is frozen beforehand
    stage2_frozen_before = all(not p.requires_grad for p in model.stage2.parameters())
    assert stage2_frozen_before, "stage2 should be frozen before unfreezing"

    unfreeze_top_layers(model)

    stage2_trainable = all(p.requires_grad for p in model.stage2.parameters())
    extra1_trainable = all(p.requires_grad for p in model.extra1.parameters())
    extra2_trainable = all(p.requires_grad for p in model.extra2.parameters())

    assert stage2_trainable, "stage2 must be trainable after unfreeze_top_layers"
    assert extra1_trainable, "extra1 must be trainable after unfreeze_top_layers"
    assert extra2_trainable, "extra2 must be trainable after unfreeze_top_layers"


# ---------------------------------------------------------------------------
# MultiBoxLoss
# ---------------------------------------------------------------------------


def test_multibox_loss_positive_anchors(loss_fn):
    """With positive anchors present both loc_loss and cls_loss must be > 0."""
    B, N, C = 2, NUM_ANCHORS, NUM_CLASSES

    pred_offsets = torch.randn(B, N, 4)
    class_logits = torch.randn(B, N, C)
    target_offsets = torch.randn(B, N, 4)

    # Mark first 8 anchors per sample as positive (class > 0)
    target_classes = torch.zeros(B, N, dtype=torch.long)
    target_classes[:, :8] = torch.randint(1, C, (B, 8))

    total_loss, loc_loss, cls_loss = loss_fn(
        pred_offsets, class_logits, target_offsets, target_classes
    )

    assert total_loss.item() > 0, "Total loss must be positive"
    assert loc_loss.item() > 0, "Loc loss must be > 0 when positive anchors exist"
    assert cls_loss.item() > 0, "Cls loss must be > 0"


def test_multibox_loss_all_negatives(loss_fn):
    """When all target classes are 0 (background) loc_loss must be exactly 0."""
    B, N, C = 2, NUM_ANCHORS, NUM_CLASSES

    pred_offsets = torch.randn(B, N, 4)
    class_logits = torch.randn(B, N, C)
    target_offsets = torch.randn(B, N, 4)
    target_classes = torch.zeros(B, N, dtype=torch.long)  # all background

    total_loss, loc_loss, cls_loss = loss_fn(
        pred_offsets, class_logits, target_offsets, target_classes
    )

    assert loc_loss.item() == 0.0, "Loc loss must be 0 when no positive anchors exist"
    assert cls_loss.item() > 0, (
        "Cls loss should be nonzero even with all negatives (HNM)"
    )


def test_multibox_loss_returns_tensors(loss_fn):
    """Loss function must return three scalar tensors."""
    B, N, C = 1, NUM_ANCHORS, NUM_CLASSES
    pred_offsets = torch.randn(B, N, 4)
    class_logits = torch.randn(B, N, C)
    target_offsets = torch.randn(B, N, 4)
    target_classes = torch.zeros(B, N, dtype=torch.long)
    target_classes[0, 0] = 1

    total_loss, loc_loss, cls_loss = loss_fn(
        pred_offsets, class_logits, target_offsets, target_classes
    )

    for t in (total_loss, loc_loss, cls_loss):
        assert isinstance(t, torch.Tensor), "Each loss component must be a Tensor"
        assert t.shape == torch.Size([]), "Each loss component must be a scalar"


def test_multibox_loss_total_equals_components(loss_fn):
    """total_loss must equal cls_loss + alpha * loc_loss (alpha=1.0)."""
    B, N, C = 2, NUM_ANCHORS, NUM_CLASSES
    pred_offsets = torch.randn(B, N, 4)
    class_logits = torch.randn(B, N, C)
    target_offsets = torch.randn(B, N, 4)
    target_classes = torch.zeros(B, N, dtype=torch.long)
    target_classes[:, :4] = torch.randint(1, C, (B, 4))

    total_loss, loc_loss, cls_loss = loss_fn(
        pred_offsets, class_logits, target_offsets, target_classes
    )

    expected = cls_loss + 1.0 * loc_loss
    assert abs(total_loss.item() - expected.item()) < 1e-5, (
        "total_loss must equal cls_loss + alpha * loc_loss"
    )


# ---------------------------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------------------------


def test_cosine_lr_starts_at_initial():
    """At epoch 0 (no warmup) LR should equal initial_lr."""
    lr = get_cosine_lr(
        initial_lr=0.001, min_lr=1e-5, epoch=0, total_epochs=10, warmup_epochs=0
    )
    assert abs(lr - 0.001) < 1e-9, f"Expected 0.001, got {lr}"


def test_cosine_lr_ends_at_min():
    """At the final epoch LR should equal min_lr."""
    lr = get_cosine_lr(
        initial_lr=0.001, min_lr=1e-5, epoch=10, total_epochs=10, warmup_epochs=0
    )
    assert abs(lr - 1e-5) < 1e-9, f"Expected min_lr=1e-5, got {lr}"


def test_cosine_lr_warmup():
    """During warmup LR should increase linearly toward initial_lr."""
    warmup_epochs = 3
    initial_lr = 0.001
    lrs = [
        get_cosine_lr(
            initial_lr=initial_lr,
            min_lr=1e-5,
            epoch=e,
            total_epochs=10,
            warmup_epochs=warmup_epochs,
        )
        for e in range(warmup_epochs)
    ]
    # LRs during warmup should be strictly increasing
    for a, b in zip(lrs, lrs[1:], strict=False):
        assert b > a, "Warmup LR should be monotonically increasing"
    # Last warmup LR should equal initial_lr
    assert abs(lrs[-1] - initial_lr) < 1e-9


def test_cosine_lr_monotone_decay():
    """After warmup the LR should be monotonically non-increasing."""
    total_epochs = 15
    warmup = 2
    lrs = [
        get_cosine_lr(0.001, 1e-5, e, total_epochs, warmup_epochs=warmup)
        for e in range(warmup, total_epochs + 1)
    ]
    for a, b in zip(lrs, lrs[1:], strict=False):
        assert b <= a + 1e-9, f"LR should be non-increasing after warmup: {a} -> {b}"


def test_cosine_lr_midpoint():
    """At the midpoint the LR should be roughly mid between initial and min."""
    initial_lr = 0.001
    min_lr = 1e-5
    total_epochs = 10
    mid_lr = get_cosine_lr(
        initial_lr, min_lr, epoch=5, total_epochs=total_epochs, warmup_epochs=0
    )
    expected = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * 0.5))
    assert abs(mid_lr - expected) < 1e-9


# ---------------------------------------------------------------------------
# Training artefacts
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent  # repo root


def test_training_metrics_recorded():
    """ml/train/metrics.json must exist with required keys and passing mAP."""
    metrics_path = ROOT / "ml" / "train" / "metrics.json"
    assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"

    data = json.loads(metrics_path.read_text())
    assert "history" in data, "metrics.json must contain 'history' key"
    assert "best_mAP_all" in data, "metrics.json must contain 'best_mAP_all' key"
    assert data["best_mAP_all"] > 0.15, (
        f"best_mAP_all={data['best_mAP_all']:.4f} should exceed 0.15"
    )

    # At least one epoch must achieve novel mAP > 0.10
    novel_maps = [e["mAP_novel_classes"] for e in data["history"]]
    assert max(novel_maps) > 0.10, (
        f"Novel class mAP never exceeded 0.10 across epochs: {novel_maps}"
    )


def test_training_log_markdown():
    """docs/training_log.md must exist and contain key headings."""
    log_path = ROOT / "docs" / "training_log.md"
    assert log_path.exists(), f"training_log.md not found at {log_path}"

    content = log_path.read_text()
    assert "Fine-Tuning" in content, "training_log.md must contain 'Fine-Tuning'"
    # Should contain an epoch row (look for epoch number pattern)
    assert "Epoch" in content or "epoch" in content, (
        "training_log.md must contain an epoch table or section"
    )


def test_checkpoint_meta_saved():
    """ml/checkpoints/best_checkpoint_meta.json must exist with correct keys."""
    ckpt_path = ROOT / "ml" / "checkpoints" / "best_checkpoint_meta.json"
    assert ckpt_path.exists(), f"best_checkpoint_meta.json not found at {ckpt_path}"

    data = json.loads(ckpt_path.read_text())
    for key in ("epoch", "best_val_map", "metrics", "config"):
        assert key in data, f"Missing key '{key}' in best_checkpoint_meta.json"

    assert data["best_val_map"] > 0.0, "best_val_map must be positive"
    assert data["metrics"]["mAP_novel_classes"] > 0.10, (
        f"Novel mAP in checkpoint must exceed 0.10, got {data['metrics']['mAP_novel_classes']}"
    )
