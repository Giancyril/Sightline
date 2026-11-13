"""
ml/tests/test_model_registry.py
-------------------------------
Unit tests for models/registry.json schema, validation rules, and loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ml.models.registry import (
    ModelEntry,
    get_model_entry,
    load_model_registry,
    resolve_registry_path,
    validate_registry_data,
)


def test_resolve_default_registry_path() -> None:
    """Ensure resolve_registry_path finds the default models/registry.json."""
    path = resolve_registry_path()
    assert path.exists()
    assert path.name == "registry.json"


def test_load_default_registry() -> None:
    """Verify loading the real models/registry.json succeeds and returns ModelEntry list."""
    entries = load_model_registry()
    assert len(entries) >= 2
    ids = [entry.id for entry in entries]
    assert "fine-tuned" in ids
    assert "coco-baseline" in ids


def test_fine_tuned_model_entry_fields() -> None:
    """Verify all required fields and specifications for the fine-tuned model."""
    entry = get_model_entry("fine-tuned")
    assert entry is not None
    assert entry.id == "fine-tuned"
    assert "Open Images" in entry.display_name
    assert entry.version.startswith("v")
    assert entry.path == "ml/export/ssd_mobilenet_v2_openimages20_weights.pth"
    assert entry.input_size == [300, 300]
    assert len(entry.class_list) == 20
    assert "Person" in entry.class_list
    assert "metrics" in entry.model_dump()
    assert entry.metrics["mAP_all"] > 0.0


def test_coco_baseline_model_entry_fields() -> None:
    """Verify all required fields and specifications for the coco-baseline model."""
    entry = get_model_entry("coco-baseline")
    assert entry is not None
    assert entry.id == "coco-baseline"
    assert "COCO" in entry.display_name
    assert entry.version.startswith("v")
    assert entry.input_size == [300, 300]
    assert len(entry.class_list) == 80
    assert "person" in entry.class_list
    assert "mAP_coco_shared" in entry.metrics


def test_validation_rejects_missing_fields() -> None:
    """Verify schema raises ValidationError when required fields are missing."""
    invalid_data = {
        "id": "missing-fields",
        # Missing display_name, version, input_size, class_list, metrics
    }
    with pytest.raises(ValidationError):
        ModelEntry.model_validate(invalid_data)


def test_validation_input_size_constraints() -> None:
    """Verify input_size must be a 2-element list of positive integers."""
    base_data = {
        "id": "test-size",
        "display_name": "Test Size Model",
        "version": "1.0",
        "path": None,
        "class_list": ["cat", "dog"],
        "metrics": {"mAP": 0.5},
    }

    # 1 element -> error
    with pytest.raises(ValidationError):
        ModelEntry.model_validate({**base_data, "input_size": [300]})

    # 3 elements -> error
    with pytest.raises(ValidationError):
        ModelEntry.model_validate({**base_data, "input_size": [300, 300, 3]})

    # Non-positive -> error
    with pytest.raises(ValidationError):
        ModelEntry.model_validate({**base_data, "input_size": [300, -1]})


def test_validation_empty_class_list() -> None:
    """Verify class_list cannot be empty."""
    invalid_data = {
        "id": "test-classes",
        "display_name": "Test Classes Model",
        "version": "1.0",
        "path": None,
        "input_size": [300, 300],
        "class_list": [],
        "metrics": {"mAP": 0.5},
    }
    with pytest.raises(ValidationError):
        ModelEntry.model_validate(invalid_data)


def test_get_nonexistent_model_returns_none() -> None:
    """Verify querying an unknown model ID returns None."""
    entry = get_model_entry("nonexistent-model-xyz")
    assert entry is None


def test_validate_registry_data_dict_and_list() -> None:
    """Verify validate_registry_data accepts both top-level list and dict with 'models' key."""
    item = {
        "id": "model-a",
        "display_name": "Model A",
        "version": "1.0",
        "path": None,
        "input_size": [300, 300],
        "class_list": ["cat"],
        "metrics": {"mAP": 0.4},
    }
    entries_list = validate_registry_data([item])
    assert len(entries_list) == 1
    assert entries_list[0].id == "model-a"

    entries_dict = validate_registry_data({"version": "1.0", "models": [item]})
    assert len(entries_dict) == 1
    assert entries_dict[0].id == "model-a"

    with pytest.raises(TypeError):
        validate_registry_data("invalid-string-data")  # type: ignore[arg-type]


def test_custom_path_not_found_raises() -> None:
    """Verify passing a nonexistent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_registry(Path("nonexistent/path/registry.json"))
