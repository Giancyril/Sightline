"""
ml/tests/test_quantization_benchmark.py
---------------------------------------
Unit tests for quantization benchmark suite.
Validates latency percentiles (p50, p95), file size calculations,
accuracy evaluation, and enforcement of stored mAP degradation threshold.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from ml.eval.benchmark_quantization import (
    DEFAULT_MAX_MAP_LOSS_THRESHOLD,
    compute_fixture_map,
    format_markdown_table,
    get_file_size_info,
    measure_cpu_latency,
    run_quantization_benchmarks,
)


def test_measure_cpu_latency() -> None:
    def dummy_runner(x: np.ndarray) -> np.ndarray:
        time.sleep(0.002)  # 2 ms mock runtime
        return x * 2.0

    dummy_input = np.ones((1, 3, 10, 10), dtype=np.float32)
    stats = measure_cpu_latency(dummy_runner, dummy_input, num_warmup=2, num_runs=5)

    assert stats["sample_count"] == 5
    assert stats["p50_latency_ms"] >= 1.0
    assert stats["p95_latency_ms"] >= stats["p50_latency_ms"]
    assert stats["throughput_fps"] > 0


def test_get_file_size_info() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_file = Path(tmp_dir) / "test.bin"
        dummy_file.write_bytes(b"x" * 2048)

        info = get_file_size_info(dummy_file)
        assert info["exists"] is True
        assert info["size_bytes"] == 2048
        assert info["size_mb"] == pytest.approx(0.0, abs=1e-3)

        missing_info = get_file_size_info(Path(tmp_dir) / "nonexistent.bin")
        assert missing_info["exists"] is False
        assert missing_info["size_bytes"] == 0


def test_compute_fixture_map() -> None:
    fixture_path = "ml/tests/fixtures/val_dump_fixture.npz"
    metrics = compute_fixture_map(fixture_path)
    assert "mAP_50" in metrics
    assert "coco_mAP" in metrics
    assert 0.0 <= metrics["mAP_50"] <= 1.0
    assert 0.0 <= metrics["coco_mAP"] <= 1.0


def test_run_quantization_benchmarks_stored_threshold() -> None:
    # Run with default stored threshold: 0.010 (1.0 mAP percentage point)
    bench = run_quantization_benchmarks(
        max_map_loss_threshold=DEFAULT_MAX_MAP_LOSS_THRESHOLD,
        num_runs=3,
    )

    assert "variants" in bench
    assert "pytorch_fp32" in bench["variants"]
    assert "summary_table" in bench

    pt_variant = bench["variants"]["pytorch_fp32"]
    assert pt_variant["map_loss"] == 0.0
    assert pt_variant["passed_threshold"] is True
    assert pt_variant["latency"]["p50_latency_ms"] > 0
    assert pt_variant["latency"]["p95_latency_ms"] > 0
    assert pt_variant["file_size"]["size_bytes"] > 0

    # Ensure all available variants respect stored degradation threshold <= 0.010
    for vname, vdata in bench["variants"].items():
        assert vdata["map_loss"] <= DEFAULT_MAX_MAP_LOSS_THRESHOLD, (
            f"{vname} exceeded threshold: {vdata['map_loss']}"
        )
        assert vdata["passed_threshold"] is True

    assert bench["all_passed_accuracy_gate"] is True


def test_stored_threshold_strict_failure() -> None:
    # Run with unrealistically tight threshold to verify failure branch
    bench = run_quantization_benchmarks(
        max_map_loss_threshold=1e-6,
        num_runs=2,
    )
    # Variants with any degradation should fail
    has_failed_variant = any(
        not v["passed_threshold"] for v in bench["variants"].values()
    )
    assert has_failed_variant is True
    assert bench["all_passed_accuracy_gate"] is False


def test_format_markdown_table() -> None:
    bench = run_quantization_benchmarks(
        max_map_loss_threshold=DEFAULT_MAX_MAP_LOSS_THRESHOLD,
        num_runs=2,
    )
    md = format_markdown_table(bench)
    assert "| Model Variant |" in md
    assert "PyTorch Float32 Baseline" in md
    assert "PASSED" in md
