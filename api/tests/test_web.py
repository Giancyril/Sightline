"""
api/tests/test_web.py
---------------------
Stage 7 unit tests: Frontend bundle verification, bundle size SLA check,
component structure, and HTML meta attributes.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

pytestmark = pytest.mark.stage7

ROOT = Path(__file__).parent.parent.parent
WEB_DIR = ROOT / "web"
DIST_DIR = WEB_DIR / "dist"


def test_web_source_components_exist():
    """Verify all required React view and component sources are present."""
    components = [
        WEB_DIR / "src" / "App.tsx",
        WEB_DIR / "src" / "api.ts",
        WEB_DIR / "src" / "types.ts",
        WEB_DIR / "src" / "components" / "DetectionCanvas.tsx",
        WEB_DIR / "src" / "components" / "WebcamCapture.tsx",
        WEB_DIR / "src" / "views" / "DetectView.tsx",
        WEB_DIR / "src" / "views" / "HistoryView.tsx",
        WEB_DIR / "src" / "views" / "ModelView.tsx",
        WEB_DIR / "src" / "views" / "ApiDocsView.tsx",
    ]
    for path in components:
        assert path.exists(), f"Missing required component source: {path.name}"


def test_web_index_html_metadata():
    """Verify index.html has accessible title and meta description."""
    index_html = WEB_DIR / "index.html"
    assert index_html.exists()

    content = index_html.read_text(encoding="utf-8")
    assert "<title>VisionSSD" in content
    assert 'name="description"' in content
    assert 'name="viewport"' in content


def test_dist_bundle_generated():
    """Verify Vite production build produces index.html and assets."""
    if not DIST_DIR.exists():
        pytest.skip("web/dist/ not found; run `npm run build` in web/ to generate")

    assert (DIST_DIR / "index.html").exists(), "dist/index.html must be generated"

    assets_dir = DIST_DIR / "assets"
    assert assets_dir.exists(), "dist/assets/ directory must exist"

    js_files = list(assets_dir.glob("*.js"))
    css_files = list(assets_dir.glob("*.css"))
    assert len(js_files) > 0, "At least one JavaScript bundle must be generated"
    assert len(css_files) > 0, "At least one CSS bundle must be generated"


def test_js_bundle_size_sla():
    """Verify gzipped JS bundle size is under 250 KB (SLA requirement)."""
    if not DIST_DIR.exists():
        pytest.skip("web/dist/ not found; run `npm run build` in web/ to generate")

    assets_dir = DIST_DIR / "assets"
    js_files = list(assets_dir.glob("*.js"))
    assert len(js_files) > 0

    total_gzipped_bytes = 0
    for js_path in js_files:
        raw_bytes = js_path.read_bytes()
        gzipped = gzip.compress(raw_bytes)
        total_gzipped_bytes += len(gzipped)

    total_kb = total_gzipped_bytes / 1024.0
    assert total_kb < 250.0, (
        f"Gzipped bundle size {total_kb:.2f} KB exceeds 250 KB SLA ceiling"
    )
