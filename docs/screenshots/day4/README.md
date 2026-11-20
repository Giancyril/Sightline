# Day 4 Screenshots — Correction Tool & Active Learning Queue

> Placeholder descriptions for the Day 4 feature screenshots.
> Replace each section with actual screenshots when the UI is running.

---

## 1. Edit Mode — Canvas with Resize Handles

**File:** `edit-mode-resize-handles.png`

Shows `DetectionCanvas` in edit mode with a bounding box selected. Eight resize handles are visible at corners and midpoints. The selected box is highlighted with a solid border; unselected boxes use a dashed border. The class label chip floats above the top-left corner.

**How to capture:**
1. Start the dev server: `pnpm dev`
2. Upload any JPEG via the Upload tab.
3. Click the **Edit** toggle in the DetectView toolbar.
4. Click a detection box to select it.
5. Screenshot at 1280×800.

---

## 2. ClassPicker Modal — Relabel Dialog

**File:** `class-picker-relabel.png`

Shows the ClassPicker modal overlay triggered by right-clicking (or pressing **L** with a box selected). Displays a grid of 20 class chips, each coloured with its design-system token. The currently assigned class is highlighted.

---

## 3. Undo/Redo — Before and After

**File:** `undo-redo-flow.png` (2-panel)

Left panel: a new bounding box has been drawn on the canvas (ADD state).  
Right panel: after pressing **Ctrl+Z**, the new box is gone and the canvas returns to its previous state.

---

## 4. Active Learning Queue — Review Panel

**File:** `review-queue-panel.png`

Shows the Review Queue tab in `DetectView`. The panel includes:
- Progress bar with pending / reviewed / corrected / skipped segments
- Status count chips
- Current item card with uncertainty score and reason callout
- Keyboard shortcut hints (R, S, [, ])
- Collapsible list drawer with status dot icons on the right

---

## 5. Corrections Export — Download Zip

**File:** `corrections-export-zip.png`

Shows the browser download dialog triggered by clicking **Export corrections (.zip)**. The zip filename is `corrections_export.zip`.

---

## 6. API Docs — Corrections Endpoints

**File:** `api-docs-corrections.png`

Shows the API Docs view with the four new Day 4 endpoints listed:
- `POST /v1/corrections`
- `GET /v1/corrections/export`
- `GET /v1/corrections/{id}`
- `DELETE /v1/corrections/{id}`

Each entry shows the method badge, path, and expandable parameter/response/snippet panels.
