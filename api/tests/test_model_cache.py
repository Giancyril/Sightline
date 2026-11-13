"""
api/tests/test_model_cache.py
-----------------------------
Unit tests for lazy model loading, in-memory caching, thread safety, and unknown IDs.
"""

from __future__ import annotations

import concurrent.futures

import pytest

from api.services.detector import ModelCache, UnknownModelError


def test_lazy_loading_and_cache_hit() -> None:
    """Ensure models are loaded on first use and cached in memory."""
    cache = ModelCache()

    # Before requesting, nothing is loaded
    assert not cache.is_loaded("fine-tuned")
    assert not cache.is_loaded("coco-baseline")
    assert cache.loaded_model_ids() == []

    # First load
    model1 = cache.get("fine-tuned")
    assert model1 is not None
    assert cache.is_loaded("fine-tuned")
    assert "fine-tuned" in cache.loaded_model_ids()

    # Second load: must be a cache hit (same instance in memory)
    model2 = cache.get("fine-tuned")
    assert model1 is model2


def test_load_coco_baseline_model() -> None:
    """Ensure coco-baseline model can be lazily loaded and cached."""
    cache = ModelCache()
    baseline = cache.get("coco-baseline")
    assert baseline.model_id == "coco-baseline"
    assert baseline.num_classes == 81
    assert cache.is_loaded("coco-baseline")


def test_unknown_model_id_raises() -> None:
    """Ensure requesting an unknown model ID raises UnknownModelError."""
    cache = ModelCache()
    with pytest.raises(UnknownModelError) as exc_info:
        cache.get("nonexistent-model-999")

    assert "nonexistent-model-999" in str(exc_info.value)
    assert exc_info.value.status_code == 404


def test_cache_clear() -> None:
    """Ensure clear() removes all cached models from memory."""
    cache = ModelCache()
    cache.get("fine-tuned")
    assert cache.is_loaded("fine-tuned")

    cache.clear()
    assert not cache.is_loaded("fine-tuned")
    assert cache.loaded_model_ids() == []


def test_thread_safe_concurrent_access() -> None:
    """Ensure multiple threads requesting the model simultaneously receive the same instance."""
    cache = ModelCache()
    results = []

    def fetch_model():
        return cache.get("fine-tuned")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_model) for _ in range(16)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert len(results) == 16
    first = results[0]
    # All 16 threads must receive the identical object identity
    for r in results:
        assert r is first
