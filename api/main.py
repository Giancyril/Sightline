"""
api/main.py
-----------
FastAPI application entry point for the Object Detection System.
Provides REST endpoints for real-time image detection, class taxonomies,
model benchmarking metadata, and service health checks.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import classes, compare, detect, model, video
from api.schemas import HealthResponse
from api.services.detector import get_detector
from api.services.video_jobs import start_cleanup_task


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm up the detector model on startup and start background cleanup."""
    get_detector()
    cleanup_task = start_cleanup_task()
    try:
        yield
    finally:
        cleanup_task.stop()


app = FastAPI(
    title="Object Detection API",
    description=(
        "Production REST API serving a fine-tuned SSD MobileNet V2 "
        "trained on a 20-class subset of the Open Images dataset."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend clients (React/Vite dev server and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints
app.include_router(detect.router)
app.include_router(compare.router)
app.include_router(classes.router)
app.include_router(model.router)
app.include_router(video.router)


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
def health() -> dict[str, Any]:
    """Health and readiness probe."""
    detector = get_detector()
    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "version": app.version,
    }
