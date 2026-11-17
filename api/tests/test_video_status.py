"""
api/tests/test_video_status.py
------------------------------
Tests for GET /v1/video/{job_id} status and results endpoint (Commit 3),
including progress, error handling, result links, and TTL cleanup of jobs
and temporary files.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from api.main import app
from api.routers.video import get_job_status
from api.services.video_jobs import (
    JobState,
    VideoCleanupTask,
    VideoJobStore,
    get_job_store,
    start_cleanup_task,
)

# ---------------------------------------------------------------------------
# Route registration in OpenAPI
# ---------------------------------------------------------------------------


def test_video_status_route_registered():
    """Verify GET /v1/video/{job_id} is mounted in the FastAPI routing table."""
    paths = list(app.openapi()["paths"].keys())
    assert "/v1/video/{job_id}" in paths


# ---------------------------------------------------------------------------
# GET /v1/video/{job_id} endpoint tests
# ---------------------------------------------------------------------------


def test_get_status_unknown_job_404():
    """Requesting an unknown job ID raises HTTP 404."""
    with pytest.raises(HTTPException) as exc_info:
        get_job_status("non-existent-job-id-999")
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_get_status_queued_job():
    """Queued job returns progress=0, state=queued, no results links."""
    store = get_job_store()
    job = store.create()
    job.state = JobState.QUEUED
    job.progress = 0.0
    store.update(job)

    try:
        data = get_job_status(job.job_id)
        assert data["job_id"] == job.job_id
        assert data["state"] == "queued"
        assert data["progress"] == 0.0
        assert data["has_results"] is False
        assert "results" not in data
    finally:
        store.delete(job.job_id)


def test_get_status_running_job():
    """Running job returns partial progress."""
    store = get_job_store()
    job = store.create()
    job.state = JobState.RUNNING
    job.progress = 0.45
    job.frames_processed = 45
    job.total_frames_to_process = 100
    store.update(job)

    try:
        data = get_job_status(job.job_id)
        assert data["state"] == "running"
        assert data["progress"] == 0.45
        assert data["frames_processed"] == 45
        assert data["total_frames_to_process"] == 100
        assert data["has_results"] is False
        assert "results" not in data
    finally:
        store.delete(job.job_id)


def test_get_status_done_job_includes_result_links():
    """Done job returns 100% progress and download URLs for outputs."""
    store = get_job_store()
    job = store.create()
    job.state = JobState.DONE
    job.progress = 1.0
    store.update(job)

    try:
        data = get_job_status(job.job_id)
        assert data["state"] == "done"
        assert data["progress"] == 1.0
        assert data["has_results"] is True
        assert "results" in data
        links = data["results"]
        assert links["annotated_video"] == f"/v1/video/{job.job_id}/annotated"
        assert links["tracks_csv"] == f"/v1/video/{job.job_id}/tracks"
        assert links["summary_json"] == f"/v1/video/{job.job_id}/summary"
    finally:
        store.delete(job.job_id)


def test_get_status_failed_job_includes_error():
    """Failed job returns state=failed and error message."""
    store = get_job_store()
    job = store.create()
    job.state = JobState.FAILED
    job.error = "Corrupted video container"
    store.update(job)

    try:
        data = get_job_status(job.job_id)
        assert data["state"] == "failed"
        assert data["error"] == "Corrupted video container"
        assert data["has_results"] is False
    finally:
        store.delete(job.job_id)


# ---------------------------------------------------------------------------
# Cleanup task & file purging tests
# ---------------------------------------------------------------------------


def test_purge_deletes_temp_files():
    """Purging an expired job also unlinks all its associated temp files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        vid_file = tmp_dir / "input.mp4"
        ann_file = tmp_dir / "annotated.mp4"
        csv_file = tmp_dir / "tracks.csv"
        json_file = tmp_dir / "summary.json"

        for f in (vid_file, ann_file, csv_file, json_file):
            f.write_text("test")

        store = VideoJobStore()
        job = store.create()
        job.video_path = vid_file
        job.annotated_video_path = ann_file
        job.tracks_csv_path = csv_file
        job.summary_json_path = json_file
        # Artificially age the job past TTL
        job.created_at = time.time() - 4000.0
        store.update(job)

        purged_count = store.purge_expired(ttl_s=3600.0)
        assert purged_count == 1
        assert store.get(job.job_id) is None
        # All temp files must be removed from disk
        assert not vid_file.exists()
        assert not ann_file.exists()
        assert not csv_file.exists()
        assert not json_file.exists()


def test_cleanup_task_lifecycle():
    """VideoCleanupTask can be started, run synchronously, and stopped."""
    store = VideoJobStore()
    task = VideoCleanupTask(store=store, interval_s=100.0, ttl_s=60.0)
    task.start()
    assert task._thread is not None and task._thread.is_alive()

    # run_once should succeed
    count = task.run_once()
    assert count == 0

    task.stop()
    assert not task._thread.is_alive()


def test_cleanup_task_auto_purges():
    """VideoCleanupTask automatically purges expired jobs via its thread."""
    with tempfile.TemporaryDirectory() as tmp:
        dummy_file = Path(tmp) / "clip.mp4"
        dummy_file.write_text("dummy")

        store = VideoJobStore()
        job = store.create()
        job.video_path = dummy_file
        job.created_at = time.time() - 100.0  # older than 1.0s TTL
        store.update(job)

        task = start_cleanup_task(store=store, interval_s=0.05, ttl_s=1.0)
        try:
            # Wait a short duration for the thread loop to trigger
            time.sleep(0.2)
            assert store.get(job.job_id) is None
            assert not dummy_file.exists()
        finally:
            task.stop()
