# Day 4 Report — Correction Tool & Active Learning Queue

> Generated: 2026-11-20

---

## Summary

Day 4 added a full human-in-the-loop correction workflow on top of the Day 1–3 detection pipeline.
Ten commits were delivered on `main` with no force-pushes, no image blobs, and no API keys committed.

---

## What Was Built

### 1. Editor State Machine (Commits 1–3)

- `web/src/lib/editorReducer.ts` — Redux-style state machine for bounding-box edits.  
  Actions: `ADD`, `MOVE`, `RESIZE`, `RELABEL`, `DELETE`, `UNDO`, `REDO`, `SET_BOXES`.
- Infinite undo/redo via past/future stacks.
- `clampBBox()` enforces `[0, 1]` bounds and 2-pixel minimum size on every mutation.
- Keyboard nudge: arrow keys shift selected box by ±0.002 (normalized) with Shift for ×5.

### 2. Canvas Edit Mode (Commits 2–4)

- `DetectionCanvas.tsx` extended with an `editMode` prop that toggles between view/edit.
- Selection, drag-to-move, 8-handle resize, draw-new-box (drag on background), relabel via `ClassPicker` modal, delete via `Backspace`/`Delete`.
- All box operations go through `editorReducer` — no direct state mutation.

### 3. Undo/Redo Wiring (Commit 5)

- `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z` wired globally in `DetectView`.
- Shortcut entries added to `ShortcutsDialog.tsx`.
- Commands added to `CommandPalette.tsx`.

### 4. Corrections API — POST & Storage (Commit 6)

**Endpoint:** `POST /v1/corrections`

**Privacy model (enforced server-side):**

| Field | Stored by default | Stored with `store_image=true` |
|---|---|---|
| `image_hash` (SHA-256) | ✅ | ✅ |
| `annotations` (COCO) | ✅ | ✅ |
| `original_predictions` | ✅ | ✅ |
| `model_id` / `model_version` | ✅ | ✅ |
| Image bytes | ❌ | ✅ (opt-in) |

Storage path: `data/corrections/<YYYY-MM-DD>/<uuid>.json`  
Root gitignore: `/data/corrections/` (committed files cannot leak).

**COCO output format:**
```json
{
  "info": { "version": "1.0", "date_created": "..." },
  "images": [{ "id": "<uuid>", "width": N, "height": M, "file_name": "<sha256>.jpg" }],
  "annotations": [{ "id": 1, "image_id": "<uuid>", "category_id": 2,
                     "bbox": [x, y, w, h], "area": A, "iscrowd": 0 }],
  "categories": [{ "id": 2, "name": "Laptop" }]
}
```
Bbox is COCO format: `[x_min_px, y_min_px, width_px, height_px]`, derived from `box_normalized` × image dimensions.

Degenerate boxes (zero width or height after pixel conversion) are rejected with HTTP 422.

### 5. Corrections Export (Commit 7)

**Endpoint:** `GET /v1/corrections/export`

Returns `application/zip` containing:
- All `*.json` COCO files from `data/corrections/`
- `manifest.json` — index of correction_id, image_hash, model_id, created_at

Also added `GET /v1/corrections/{id}` (retrieve single document) and round-trip test.

### 6. Uncertainty Ranking (Commit 8)

**File:** `ml/eval/active_learning.py`

**Compound uncertainty score formula:**

```
score = w_thresh × threshold_proximity + w_disagree × disagreement_rate
```

Where:
- `threshold_proximity = Σ exp(-δ / 0.05)` for each detection near its per-class threshold (δ = |conf − thresh|)
- `disagreement_rate = unmatched_primary + unmatched_secondary` / `max(total_primary, total_secondary, 1)`, using bipartite IoU matching from `ml/eval/iou_metrics.py`
- Default weights: `w_thresh = 0.6`, `w_disagree = 0.4`

Images with score > 0.0 are ranked descending. Equal scores sort by image hash (stable).

### 7. Review Queue UI (Commit 9)

**New component:** `web/src/components/detect/ReviewQueue.tsx`

Features:
- Queue sorted by compound uncertainty score (highest uncertainty first)
- Per-item uncertainty reason callout (borderline scores, model disagreements)
- Status tracking: `pending` → `reviewed` / `skipped` / `corrected`
- Progress counter with colour-coded breakdown bar
- Keyboard shortcuts: **R** = mark reviewed, **S** = skip, **[** = prev, **]** = next
- Collapsible list drawer with status dot indicators
- Full keyboard accessibility (visible focus, aria-labels)

**New view tab:** `DetectView.tsx` gains a `"queue"` mode tab.

**API client additions:** `saveCorrection()`, `getCorrectionsExportUrl()` in `web/src/api.ts`.

**New types:** `CorrectionAnnotation`, `CorrectionCreateRequest`, `CorrectionResponse`, `ReviewStatus`, `ReviewQueueItem` in `web/src/types.ts`.

---

## Privacy Notes

> [!IMPORTANT]
> **Zero-storage default.** By default, `POST /v1/corrections` stores ONLY:
> - The SHA-256 hex hash of the image (not the image itself)
> - Annotation data (class IDs, bounding boxes, model predictions)
> - Model ID, model version, and correction UUID
>
> Image bytes are NEVER written to disk unless the caller explicitly sets `store_image: true`.

- **Gitignore protection:** `/data/corrections/` is rooted in `.gitignore` — correction data can never accidentally be committed.
- **Deletion API:** `DELETE /v1/corrections/{id}` removes the COCO JSON and any stored image file.
- **No session storage:** The correction queue in the UI uses ephemeral in-memory state; no localStorage, no IndexedDB, no cookies.
- **Image hash only:** SHA-256 is a one-way function. The original image cannot be reconstructed from the hash.

---

## Uncertainty Score Formula Detail

```
threshold_proximity(image) = Σ_{d ∈ detections} exp(−|conf_d − thresh_d| / 0.05)
```

- `thresh_d` = per-class threshold from `thresholds.json` (Day 3), or global 0.25 fallback.
- Exponential decay with σ = 0.05 gives full weight to conf within 5pp of threshold, negligible weight beyond 20pp.

```
disagreement_rate(image) = (unmatched_primary + unmatched_secondary) / max(N_primary, N_secondary, 1)
```

- Bipartite matching uses the same `matchDetections()` function from `web/src/lib/iouMatching.ts` (TypeScript, IoU ≥ 0.5) and `ml/eval/iou_metrics.py` (Python, IoU ≥ 0.5).
- A detection is "unmatched" if no partner in the other model's output has IoU ≥ 0.5.

```
compound_score = 0.6 × threshold_proximity + 0.4 × disagreement_rate
```

---

## Screenshots

See `docs/screenshots/day4/README.md` for placeholder descriptions.

---

## 5 Biggest Remaining Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **No real `POST /v1/corrections` call from queue UI** — the ReviewQueue uses hardcoded sample data (`createDefaultReviewQueue()`), not live detections from the detect tab. The `saveCorrection()` API client exists but is not wired to a live image+detection payload in the queue flow. | High | Wire `DetectView` to pass real `detections` + `detectionResponse` state into ReviewQueue; call `saveCorrection()` with the actual SHA-256 of the uploaded image. |
| 2 | **No browser E2E test** — Playwright is not installed; `editorFlow.test.ts` tests the state machine headlessly but cannot verify canvas rendering, mouse drag interactions, or the full correction submit flow. | High | Add Playwright to `devDependencies` and write canvas interaction tests with `page.mouse.move/click/drag` in CI. |
| 3 | **No queue persistence across sessions** — `ReviewQueueItem[]` state lives in React component memory. Closing the tab resets all review statuses. | Medium | Persist queue state to `localStorage` keyed by image_hash; rehydrate on mount. |
| 4 | **No UI delete button for stored corrections** — `DELETE /v1/corrections/{id}` exists in the API but no button in the correction flow or queue drawer calls it. | Medium | Add a "Delete correction" action in the queue item detail pane, calling the delete endpoint and removing the item from local state. |
| 5 | **Uncertainty weights are hard-coded** — `w_thresh = 0.6`, `w_disagree = 0.4` are constants in `active_learning.py`. Practitioners cannot tune them without editing source. | Low | Expose weights as query parameters on a future `GET /v1/active-learning/rank` endpoint, or as a user-configurable settings panel. |
