"""
api/tests/test_video_jobs.py
----------------------------
Unit tests for video upload validation and the VideoJobStore.
Covers: validation errors, job state transitions, store CRUD, TTL cleanup.
"""

from __future__ import annotations

import struct
import time
import uuid

import pytest

from api.services.video_jobs import (
    JOB_TTL_S,
    MAX_VIDEO_SIZE_BYTES,
    JobState,
    VideoJob,
    VideoJobStore,
    VideoValidationError,
    validate_video_bytes,
)

# ---------------------------------------------------------------------------
# Helpers – synthetic byte signatures
# ---------------------------------------------------------------------------


def _make_mp4_bytes(extra: bytes = b"\x00" * 100) -> bytes:
    """Return minimal bytes that look like an MP4 (ftyp box at offset 4)."""
    # size(4) + "ftyp" + "isom" + version(4) + extra
    box = struct.pack(">I", 20) + b"ftypisom" + b"\x00" * 8
    return box + extra


def _make_webm_bytes(extra: bytes = b"\x00" * 100) -> bytes:
    """Return minimal bytes that look like a WebM/MKV (EBML magic)."""
    return b"\x1a\x45\xdf\xa3" + extra


# ---------------------------------------------------------------------------
# 1. validate_video_bytes – happy paths
# ---------------------------------------------------------------------------


def test_validate_mp4_bytes_accepted():
    data = _make_mp4_bytes()
    mime = validate_video_bytes(data, "clip.mp4")
    assert mime == "video/mp4"


def test_validate_webm_bytes_accepted():
    data = _make_webm_bytes()
    mime = validate_video_bytes(data, "clip.webm")
    assert mime == "video/webm"


# ---------------------------------------------------------------------------
# 2. validate_video_bytes – error paths
# ---------------------------------------------------------------------------


def test_validate_empty_raises_400():
    with pytest.raises(VideoValidationError) as exc:
        validate_video_bytes(b"", "empty.mp4")
    assert exc.value.status_code == 400


def test_validate_too_large_raises_413():
    big = b"\xff" * (MAX_VIDEO_SIZE_BYTES + 1)
    with pytest.raises(VideoValidationError) as exc:
        validate_video_bytes(big, "big.mp4")
    assert exc.value.status_code == 413


def test_validate_wrong_extension_raises_415():
    data = _make_mp4_bytes()
    with pytest.raises(VideoValidationError) as exc:
        validate_video_bytes(data, "video.avi")
    assert exc.value.status_code == 415


def test_validate_bad_content_mp4_extension_raises_415():
    with pytest.raises(VideoValidationError) as exc:
        validate_video_bytes(b"This is plain text", "clip.mp4")
    assert exc.value.status_code == 415


def test_validate_bad_content_webm_extension_raises_415():
    with pytest.raises(VideoValidationError) as exc:
        validate_video_bytes(b"Not webm content!!!", "clip.webm")
    assert exc.value.status_code == 415


# ---------------------------------------------------------------------------
# 3. VideoJob dataclass
# ---------------------------------------------------------------------------


def test_video_job_defaults():
    job = VideoJob()
    assert job.state == JobState.QUEUED
    assert job.progress == 0.0
    assert job.error is None
    assert len(job.job_id) > 0


def test_video_job_to_status_dict():
    job = VideoJob()
    d = job.to_status_dict()
    assert d["state"] == "queued"
    assert d["progress"] == 0.0
    assert d["has_results"] is False


def test_video_job_done_has_results():
    job = VideoJob()
    job.state = JobState.DONE
    d = job.to_status_dict()
    assert d["has_results"] is True


def test_video_job_touch_updates_timestamp():
    job = VideoJob()
    t0 = job.updated_at
    time.sleep(0.01)
    job.touch()
    assert job.updated_at > t0


# ---------------------------------------------------------------------------
# 4. VideoJobStore CRUD
# ---------------------------------------------------------------------------


def test_store_create_and_get():
    store = VideoJobStore()
    job = store.create()
    fetched = store.get(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id


def test_store_get_unknown_returns_none():
    store = VideoJobStore()
    assert store.get(str(uuid.uuid4())) is None


def test_store_update():
    store = VideoJobStore()
    job = store.create()
    job.state = JobState.RUNNING
    store.update(job)
    fetched = store.get(job.job_id)
    assert fetched.state == JobState.RUNNING


def test_store_delete():
    store = VideoJobStore()
    job = store.create()
    store.delete(job.job_id)
    assert store.get(job.job_id) is None


def test_store_all_jobs():
    store = VideoJobStore()
    j1 = store.create()
    j2 = store.create()
    ids = {j.job_id for j in store.all_jobs()}
    assert j1.job_id in ids
    assert j2.job_id in ids


# ---------------------------------------------------------------------------
# 5. TTL cleanup
# ---------------------------------------------------------------------------


def test_purge_expired_removes_old_jobs():
    store = VideoJobStore()
    old_job = store.create()
    # Backdate creation time beyond TTL
    old_job.created_at = time.time() - JOB_TTL_S - 10
    store.update(old_job)

    fresh_job = store.create()

    removed = store.purge_expired(ttl_s=JOB_TTL_S)
    assert removed >= 1
    assert store.get(old_job.job_id) is None
    assert store.get(fresh_job.job_id) is not None


def test_purge_does_not_remove_fresh_jobs():
    store = VideoJobStore()
    job = store.create()
    removed = store.purge_expired(ttl_s=JOB_TTL_S)
    assert removed == 0
    assert store.get(job.job_id) is not None
