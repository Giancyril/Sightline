"""
ml/train/colab_runner.py
------------------------
Standalone Google Colab / Kaggle notebook harness for GPU fine-tuning.
Handles GPU validation, virtual environment dependencies, YAML config injection,
resumable checkpointing, and metrics extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from ml.train.train import train


def setup_notebook_environment() -> None:
    """Validate GPU availability and environment prerequisites."""
    print("=" * 60)
    print("  Object Detection System: Colab/Kaggle GPU Harness")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    else:
        print(
            "Note: Running in CPU mode (GPU recommended for full 15-epoch fine-tuning)."
        )
    print("=" * 60)


def run_colab_training(epochs: int = 15) -> None:
    setup_notebook_environment()
    config_path = Path("ml/train/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    print(f"Starting training run with config: {config_path} ({epochs} epochs)")
    train(config_path=config_path, max_epochs=epochs)


if __name__ == "__main__":
    run_colab_training()
