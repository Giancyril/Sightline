"""api/main.py — FastAPI application entry point (skeleton)."""

from fastapi import FastAPI

app = FastAPI(
    title="Object Detection API",
    description="Fine-tuned SSD object detection service.",
    version="0.1.0",
)


@app.get("/healthz", tags=["health"])
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
